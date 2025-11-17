# topopt_compare_Q4_T3.py
# Topology optimization: compare Q4 (rectangular) vs T3 (triangular split of quads)
# SIMP + sensitivity filtering + Optimality Criteria
#
# Requires: numpy, scipy, matplotlib, pillow
# Example run: python topopt_compare_Q4_T3.py

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import time

# Material & problem parameters
E0 = 1.0
Emin = 1e-9
nu = 0.3

def D_matrix(E=E0, nu=nu, plane_stress=True):
    if plane_stress:
        coef = E / (1 - nu**2)
        D = coef * np.array([[1, nu, 0],
                             [nu, 1, 0],
                             [0, 0, (1-nu)/2]])
    else:
        coef = E / ((1+nu)*(1-2*nu))
        D = coef * np.array([[1-nu, nu, 0],
                             [nu, 1-nu, 0],
                             [0, 0, (1-2*nu)/2]])
    return D

def KE_quad(E=1.0):
    # compact Sigmund-like 8x8 element stiffness (normalized)
    k = np.array([
        [ 0.6667, -0.3333, -0.3333, -0.0000, -0.3333,  0.0000,  0.0000,  0.0000],
        [-0.3333,  0.6667,  0.0000,  0.0000,  0.0000, -0.3333, -0.0000, -0.3333],
        [-0.3333,  0.0000,  0.6667, -0.3333,  0.0000, -0.0000, -0.3333,  0.0000],
        [-0.0000,  0.0000, -0.3333,  0.6667,  0.0000,  0.0000, -0.3333, -0.3333],
        [-0.3333,  0.0000,  0.0000,  0.0000,  0.6667, -0.3333, -0.3333, -0.0000],
        [ 0.0000, -0.3333, -0.0000,  0.0000, -0.3333,  0.6667,  0.0000, -0.3333],
        [ 0.0000, -0.0000, -0.3333, -0.3333, -0.3333,  0.0000,  0.6667,  0.0000],
        [ 0.0000, -0.3333,  0.0000, -0.3333, -0.0000, -0.3333,  0.0000,  0.6667]
    ])
    return E * k

def KE_tri(coords, D):
    # Linear triangular element stiffness computed using B-matrix
    x1,y1 = coords[0]
    x2,y2 = coords[1]
    x3,y3 = coords[2]
    A = 0.5 * ((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
    Aabs = abs(A)
    if Aabs < 1e-12:
        return np.zeros((6,6)), 0.0
    b1 = y2 - y3; b2 = y3 - y1; b3 = y1 - y2
    c1 = x3 - x2; c2 = x1 - x3; c3 = x2 - x1
    B = (1.0/(2*Aabs)) * np.array([[b1, 0, b2, 0, b3, 0],
                                   [0, c1, 0, c2, 0, c3],
                                   [c1, b1, c2, b2, c3, b3]])
    ke = (B.T @ D @ B) * Aabs
    return ke, Aabs

def mesh_Q4(nelx, nely, lx=1.0, ly=1.0):
    nx = nelx+1; ny = nely+1
    x_coords = np.linspace(0,lx,nx)
    y_coords = np.linspace(0,ly,ny)
    xv, yv = np.meshgrid(x_coords, y_coords)
    coords = np.vstack([xv.ravel(), yv.ravel()]).T
    return coords

def mesh_T3_from_Q4(nelx, nely, lx=1.0, ly=1.0):
    nx = nelx+1; ny = nely+1
    x_coords = np.linspace(0,lx,nx)
    y_coords = np.linspace(0,ly,ny)
    xv, yv = np.meshgrid(x_coords, y_coords)
    coords = np.vstack([xv.ravel(), yv.ravel()]).T
    tri_elems = []
    for i in range(nelx):
        for j in range(nely):
            n1 = j*(nelx+1) + i
            n2 = (j+1)*(nelx+1) + i
            n3 = (j+1)*(nelx+1) + i+1
            n4 = j*(nelx+1) + i+1
            tri_elems.append([n1, n2, n3])
            tri_elems.append([n1, n3, n4])
    return coords, np.array(tri_elems, dtype=int)

def build_Q4_edof(nelx, nely):
    nele = nelx * nely
    edofMat = np.zeros((nele,8), dtype=int)
    el = 0
    for i in range(nelx):
        for j in range(nely):
            n1 = j*(nelx+1) + i
            n2 = (j+1)*(nelx+1) + i
            n3 = (j+1)*(nelx+1) + i+1
            n4 = j*(nelx+1) + i+1
            edofMat[el,:] = np.array([2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n3, 2*n3+1, 2*n4, 2*n4+1])
            el += 1
    return edofMat

# === Q4 optimization ===
def topopt_Q4(nelx=40, nely=20, volfrac=0.4, penal=3.0, rmin=1.5, ft=1, 
              maxiter=80, tol=1e-3, display_interval=10):
    coords = mesh_Q4(nelx, nely)
    nnodes = coords.shape[0]
    ndof = 2*nnodes
    nele = nelx * nely
    edofMat = build_Q4_edof(nelx, nely)
    KE = KE_quad()
    # Loads and supports (MBB style)
    F = np.zeros((ndof,1))
    mid_y = nely//2
    right_node = mid_y*(nelx+1) + nelx
    F[2*right_node+1,0] = -1.0
    left_node_bottom = 0
    left_node_mid = mid_y*(nelx+1)
    fixeddofs = np.array([2*left_node_bottom,2*left_node_bottom+1, 2*left_node_mid,2*left_node_mid+1])
    alldofs = np.arange(ndof)
    freedofs = np.setdiff1d(alldofs, fixeddofs)
    # Filter assembly
    H = np.zeros((nele, nele)); Hs = np.zeros(nele)
    elcoords = []
    for i in range(nelx):
        for j in range(nely):
            elcoords.append((i,j))
    for i in range(nele):
        xi, yi = elcoords[i]
        for j in range(nele):
            xj, yj = elcoords[j]
            dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            if dist <= rmin:
                w = rmin - dist
                H[i,j] = w
                Hs[i] += w
    Hs += 1e-12
    # initialize densities
    x = volfrac * np.ones(nele); xPhys = x.copy()
    change = 1.0; loop = 0
    KE_flat = KE.reshape(64)
    KE_tile = np.tile(KE_flat, nele)
    iK = np.kron(edofMat, np.ones((8,1))).flatten().astype(int)
    jK = np.kron(edofMat, np.ones((1,8))).flatten().astype(int)
    while change > tol and loop < maxiter:
        loop += 1
        s_factor = np.repeat(Emin + xPhys**penal * (E0 - Emin), 64)
        sK = s_factor * KE_tile
        K = sp.coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        K_ff = K[freedofs[:,None], freedofs]
        Ff = F[freedofs,0]
        U = np.zeros((ndof,1))
        U_free = spla.spsolve(K_ff.tocsr(), Ff)
        U[freedofs,0] = U_free
        ce = np.zeros(nele)
        for el in range(nele):
            ue = U[edofMat[el,:],0]
            ce[el] = ue.T @ KE @ ue
        obj = ((Emin + xPhys**penal*(E0-Emin)) * ce).sum()
        dc = -penal * xPhys**(penal-1) * (E0 - Emin) * ce
        dv = np.ones(nele)
        if ft == 1:
            dc = (H @ (x * dc)) / (Hs * np.maximum(1e-9, x))
        elif ft == 2:
            x = (H @ x) / Hs
        # OC update
        l1=0; l2=1e9; move=0.2
        while (l2-l1)/(l1+l2+1e-9) > 1e-3:
            lmid = 0.5*(l2+l1)
            B = np.maximum(0.0, np.minimum(1.0, x * np.sqrt(np.maximum(0.0, -dc/(dv*lmid)))))
            if B.mean() - volfrac > 0:
                l1 = lmid
            else:
                l2 = lmid
        xnew = B
        xnew = np.maximum(x-move, np.minimum(x+move, xnew))
        if ft == 1:
            xPhys = xnew.copy()
        elif ft == 2:
            xPhys = (H @ xnew) / Hs
        change = np.max(np.abs(xnew-x))
        x = xnew.copy()
        if loop % display_interval == 0 or loop==1 or change < tol or loop==maxiter:
            print(f"[Q4] Iter {loop}: Obj {obj:.4f}, Vol {xPhys.mean():.3f}, Change {change:.4f}")
    density = xPhys.reshape((nelx, nely), order='F')
    density_plot = np.flipud(density.T)
    return density_plot, obj, loop, coords, edofMat, U

# === T3 optimization (triangles from quad split) ===
def topopt_T3(nelx=40, nely=20, volfrac=0.4, penal=3.0, rmin=1.5, ft=1, 
               maxiter=80, tol=1e-3, display_interval=10):
    coords, tri = mesh_T3_from_Q4(nelx, nely)
    nnodes = coords.shape[0]; ndof = 2*nnodes
    nele = tri.shape[0]
    edofMat = np.zeros((nele,6), dtype=int)
    for el in range(nele):
        nodes = tri[el]
        edofMat[el,:] = np.array([2*nodes[0],2*nodes[0]+1, 2*nodes[1],2*nodes[1]+1, 2*nodes[2],2*nodes[2]+1])
    KE_list = []; area_list = []
    for el in range(nele):
        nodes = tri[el]
        coords_el = coords[nodes]
        ke, A = KE_tri(coords_el, D_matrix())
        KE_list.append(ke)
        area_list.append(A)
    KE_list = np.array(KE_list)
    F = np.zeros((ndof,1))
    mid_y = nely//2
    right_node = mid_y*(nelx+1) + nelx
    F[2*right_node+1,0] = -1.0
    left_node_bottom = 0
    left_node_mid = mid_y*(nelx+1)
    fixeddofs = np.array([2*left_node_bottom,2*left_node_bottom+1, 2*left_node_mid,2*left_node_mid+1])
    alldofs = np.arange(ndof)
    freedofs = np.setdiff1d(alldofs, fixeddofs)
    # Filter (map triangle elements to quad coordinates: two triangles per quad)
    elcoords = []
    for i in range(nelx):
        for j in range(nely):
            elcoords.append((i,j)); elcoords.append((i,j))
    elcoords = elcoords[:nele]
    H = np.zeros((nele, nele)); Hs = np.zeros(nele)
    for i in range(nele):
        xi, yi = elcoords[i]
        for j in range(nele):
            xj, yj = elcoords[j]
            dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            if dist <= rmin:
                w = rmin - dist
                H[i,j] = w
                Hs[i] += w
    Hs += 1e-12
    x = volfrac * np.ones(nele); xPhys = x.copy()
    change = 1.0; loop = 0
    iK = np.kron(edofMat, np.ones((6,1))).flatten().astype(int)
    jK = np.kron(edofMat, np.ones((1,6))).flatten().astype(int)
    while change > tol and loop < maxiter:
        loop += 1
        sK_vals = np.zeros(nele*36)
        for el in range(nele):
            s = (Emin + xPhys[el]**penal * (E0-Emin))
            sK_vals[el*36:(el+1)*36] = (s * KE_list[el].reshape(36))
        K = sp.coo_matrix((sK_vals, (iK, jK)), shape=(ndof, ndof)).tocsc()
        K_ff = K[freedofs[:,None], freedofs]
        Ff = F[freedofs,0]
        U = np.zeros((ndof,1))
        U_free = spla.spsolve(K_ff.tocsr(), Ff)
        U[freedofs,0] = U_free
        ce = np.zeros(nele)
        for el in range(nele):
            ue = U[edofMat[el,:],0]
            ce[el] = ue.T @ KE_list[el] @ ue
        obj = ((Emin + xPhys**penal*(E0-Emin)) * ce).sum()
        dc = -penal * xPhys**(penal-1) * (E0 - Emin) * ce
        dv = np.ones(nele)
        if ft == 1:
            dc = (H @ (x * dc)) / (Hs * np.maximum(1e-9, x))
        elif ft == 2:
            x = (H @ x) / Hs
        l1=0; l2=1e9; move=0.2
        while (l2-l1)/(l1+l2+1e-9) > 1e-3:
            lmid = 0.5*(l2+l1)
            B = np.maximum(0.0, np.minimum(1.0, x * np.sqrt(np.maximum(0.0, -dc/(dv*lmid)))))
            if B.mean() - volfrac > 0:
                l1 = lmid
            else:
                l2 = lmid
        xnew = B
        xnew = np.maximum(x-move, np.minimum(x+move, xnew))
        if ft == 1:
            xPhys = xnew.copy()
        elif ft == 2:
            xPhys = (H @ xnew) / Hs
        change = np.max(np.abs(xnew-x))
        x = xnew.copy()
        if loop % display_interval == 0 or loop==1 or change < tol or loop==maxiter:
            print(f"[T3] Iter {loop}: Obj {obj:.4f}, Vol {xPhys.mean():.3f}, Change {change:.4f}")
    # Map triangular densities to quad grid by averaging two triangles per quad
    quad_nele = nelx * nely
    quad_density = np.zeros(quad_nele)
    for q in range(quad_nele):
        t1 = 2*q; t2 = 2*q+1
        if t2 < nele:
            quad_density[q] = 0.5*(xPhys[t1] + xPhys[t2])
        else:
            quad_density[q] = xPhys[t1]
    density = quad_density.reshape((nelx, nely), order='F')
    density_plot = np.flipud(density.T)
    return density_plot, obj, loop, coords, edofMat, U

# === Run comparison (tunable mesh size) ===
if __name__ == "__main__":
    nelx = 40; nely = 20   # change to larger values for higher resolution
    volfrac = 0.9; penal=3.0; rmin=1.5
    start = time.time()
    q4_density, q4_obj, q4_iters, q4_coords, q4_edof, q4_U = topopt_Q4(nelx=nelx, nely=nely, volfrac=volfrac, penal=penal, rmin=rmin, maxiter=80, display_interval=20)
    t3_density, t3_obj, t3_iters, t3_coords, t3_edof, t3_U = topopt_T3(nelx=nelx, nely=nely, volfrac=volfrac, penal=penal, rmin=rmin, maxiter=80, display_interval=20)
    end = time.time()
    print("Total runtime (s):", end-start)
    # Plot/save comparison
    fig, axes = plt.subplots(1,2, figsize=(12,5))
    axes[0].imshow(q4_density, cmap=cm.gray, vmin=0, vmax=1); axes[0].set_title(f"Q4 density - Obj {q4_obj:.4f} iters {q4_iters}"); axes[0].axis('off')
    axes[1].imshow(t3_density, cmap=cm.gray, vmin=0, vmax=1); axes[1].set_title(f"T3 density mapped - Obj {t3_obj:.4f} iters {t3_iters}"); axes[1].axis('off')
    outpath = "topopt_Q4_vs_T3.png"
    plt.tight_layout(); plt.savefig(outpath, dpi=200); plt.close()
    Image.fromarray((q4_density*255).astype(np.uint8)).convert("RGB").save("topopt_Q4.png")
    Image.fromarray((t3_density*255).astype(np.uint8)).convert("RGB").save("topopt_T3.png")
    print("Saved images: topopt_Q4_vs_T3.png, topopt_Q4.png, topopt_T3.png")
