# main.py
# Q4 vs T3 comparison with Q4 variants:
#   - isoparametric (2x2 Gauss)
#   - reduced (1x1 Gauss)
#   - sigmund (compact 8x8 SIMP KE)
#
# T3 implementation is left unchanged.

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import time

# ------------------------------
# Material parameters
# ------------------------------
E0 = 1.0
Emin = 1e-9
nu = 0.3
thickness = 1.0

# ------------------------------
# Constitutive matrix (plane stress)
# ------------------------------
def D_matrix(E=E0, nu=nu):
    coef = E / (1 - nu**2)
    return coef * np.array([[1, nu, 0],
                            [nu, 1, 0],
                            [0, 0, (1-nu)/2]])

# ------------------------------
# Sigmund compact KE (8x8) used in many SIMP codes
# ------------------------------
def KE_sigmund(E=1.0):
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
    return E * (0.5*(k + k.T))

# ------------------------------
# Q4 isoparametric (2x2 Gauss)
# ------------------------------
def KE_Q4_isoparametric(dx, dy, E=1.0, nu=0.3, thickness=1.0):
    D = D_matrix(E, nu)
    gp = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    ke = np.zeros((8,8))
    J = np.array([[dx/2, 0.0],
                  [0.0,  dy/2]])
    detJ = np.linalg.det(J)
    invJ = np.linalg.inv(J)
    for xi in gp:
        for eta in gp:
            dN_dxi = np.array([ -(1-eta)/4,  (1-eta)/4,  (1+eta)/4, -(1+eta)/4 ])
            dN_deta= np.array([ -(1-xi)/4,  -(1+xi)/4,  (1+xi)/4,   (1-xi)/4 ])
            dN_dx = invJ[0,0]*dN_dxi + invJ[0,1]*dN_deta
            dN_dy = invJ[1,0]*dN_dxi + invJ[1,1]*dN_deta
            B = np.zeros((3,8))
            for i in range(4):
                B[0,2*i]   = dN_dx[i]
                B[1,2*i+1] = dN_dy[i]
                B[2,2*i]   = dN_dy[i]
                B[2,2*i+1] = dN_dx[i]
            ke += (B.T @ D @ B) * detJ * thickness
    return 0.5*(ke + ke.T)

# ------------------------------
# Q4 reduced integration (1x1 Gauss)
# ------------------------------
def KE_Q4_reduced(dx, dy, E=1.0, nu=0.3, thickness=1.0):
    D = D_matrix(E, nu)
    # single Gauss point at (0,0) with weight 4 (since integral over [-1,1]^2)
    xi = 0.0; eta = 0.0
    dN_dxi = np.array([ -(1-eta)/4,  (1-eta)/4,  (1+eta)/4, -(1+eta)/4 ])
    dN_deta= np.array([ -(1-xi)/4,  -(1+xi)/4,  (1+xi)/4,   (1-xi)/4 ])
    J = np.array([[dx/2, 0.0],
                  [0.0,  dy/2]])
    detJ = np.linalg.det(J)
    invJ = np.linalg.inv(J)
    dN_dx = invJ[0,0]*dN_dxi + invJ[0,1]*dN_deta
    dN_dy = invJ[1,0]*dN_dxi + invJ[1,1]*dN_deta
    B = np.zeros((3,8))
    for i in range(4):
        B[0,2*i]   = dN_dx[i]
        B[1,2*i+1] = dN_dy[i]
        B[2,2*i]   = dN_dy[i]
        B[2,2*i+1] = dN_dx[i]
    weight = 4.0
    ke = (B.T @ D @ B) * detJ * thickness * weight
    return 0.5*(ke + ke.T)

# ------------------------------
# Mesh / edof helpers
# ------------------------------
def mesh_Q4(nelx, nely, lx=1.0, ly=1.0):
    nx = nelx + 1; ny = nely + 1
    xv, yv = np.meshgrid(np.linspace(0,lx,nx), np.linspace(0,ly,ny))
    coords = np.vstack([xv.ravel(), yv.ravel()]).T
    return coords

def build_Q4_edof(nelx, nely):
    nele = nelx * nely
    edofMat = np.zeros((nele,8), dtype=int)
    el = 0
    for i in range(nelx):
        for j in range(nely):
            n1 = j*(nelx+1) + i
            n2 = (j+1)*(nelx+1) + i
            n3 = (j+1)*(nelx+1) + i + 1
            n4 = j*(nelx+1) + i + 1
            edofMat[el,:] = [2*n1,2*n1+1, 2*n2,2*n2+1, 2*n3,2*n3+1, 2*n4,2*n4+1]
            el += 1
    return edofMat

# ------------------------------
# T3 mesh (unchanged)
# ------------------------------
def mesh_T3_from_Q4(nelx, nely, lx=1.0, ly=1.0):
    nx = nelx+1; ny = nely+1
    xv, yv = np.meshgrid(np.linspace(0,lx,nx), np.linspace(0,ly,ny))
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

# ------------------------------
# T3 element KE (unchanged)
# ------------------------------
def KE_tri(coords_el, D):
    x1,y1 = coords_el[0]; x2,y2 = coords_el[1]; x3,y3 = coords_el[2]
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

# ------------------------------
# Q4 topology optimization (with selectable KE mode)
# ------------------------------
def topopt_Q4(nelx=40, nely=20, volfrac=0.4, penal=3.0, rmin=1.5,
              maxiter=80, tol=1e-3, display_interval=10, q4_mode='reduced'):
    """
    q4_mode: 'isoparametric' | 'reduced' | 'sigmund'
    """

    lx = 1.0; ly = 1.0
    dx = lx / nelx
    dy = ly / nely

    coords = mesh_Q4(nelx, nely, lx, ly)
    nnodes = coords.shape[0]
    ndof = 2*nnodes
    nele = nelx * nely
    edofMat = build_Q4_edof(nelx, nely)

    # choose KE0 depending on mode
    if q4_mode == 'isoparametric':
        KE0 = KE_Q4_isoparametric(dx, dy, E=1.0, nu=nu, thickness=thickness)
    elif q4_mode == 'reduced':
        KE0 = KE_Q4_reduced(dx, dy, E=1.0, nu=nu, thickness=thickness)
    elif q4_mode == 'sigmund':
        # Sigmund compact KE scaled by element area (common approach)
        KE0 = KE_sigmund(1.0) * (dx * dy * thickness)
    else:
        raise ValueError("q4_mode must be 'isoparametric','reduced' or 'sigmund'")

    # Loads and supports (MBB-like)
    F = np.zeros((ndof,1))
    mid_y = nely//2
    right_node = mid_y*(nelx+1) + nelx
    F[2*right_node+1,0] = -1.0

    left_node_bottom = 0
    left_node_mid = mid_y*(nelx+1)
    fixeddofs = np.array([2*left_node_bottom,2*left_node_bottom+1, 2*left_node_mid, 2*left_node_mid+1])
    alldofs = np.arange(ndof)
    freedofs = np.setdiff1d(alldofs, fixeddofs)

    # filter (element-index based like original; rmin is in element units here)
    H = np.zeros((nele, nele)); Hs = np.zeros(nele)
    elcoords = [(i,j) for i in range(nelx) for j in range(nely)]
    for e1 in range(nele):
        x1,y1 = elcoords[e1]
        for e2 in range(nele):
            x2,y2 = elcoords[e2]
            dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            if dist <= rmin:
                w = rmin - dist
                H[e1,e2] = w
                Hs[e1] += w
    Hs += 1e-12

    # FE assembly helpers
    KE_flat = KE0.reshape(64)
    KE_tile = np.tile(KE_flat, nele)
    iK = np.kron(edofMat, np.ones((8,1))).flatten().astype(int)
    jK = np.kron(edofMat, np.ones((1,8))).flatten().astype(int)

    # initialize
    x = volfrac*np.ones(nele); xPhys = x.copy()
    change = 1.0; loop = 0
    move = 0.2

    history = {'obj':[], 'vol':[], 'change':[]}

    while change > tol and loop < maxiter:
        loop += 1

        sE = Emin + xPhys**penal * (E0 - Emin)
        sK_vals = np.repeat(sE, 64) * KE_tile
        K = sp.coo_matrix((sK_vals, (iK, jK)), shape=(ndof, ndof)).tocsc()

        Kff = K[freedofs[:,None], freedofs]
        Ff = F[freedofs,0]
        U = np.zeros((ndof,1))
        U_free = spla.spsolve(Kff, Ff)
        U[freedofs,0] = U_free

        # element energies
        ce = np.zeros(nele)
        for el in range(nele):
            ue = U[edofMat[el,:],0]
            ce[el] = float(ue.T @ KE0 @ ue)

        obj = np.sum(sE * ce)
        dc = -penal * xPhys**(penal-1) * (E0 - Emin) * ce
        dv = np.ones(nele)

        # filter sensitivities
        dc = (H @ (x * dc)) / (Hs * np.maximum(1e-9, x))

        # OC update
        l1 = 0.0; l2 = 1e9
        while (l2 - l1) / (l1 + l2 + 1e-12) > 1e-3:
            lmid = 0.5 * (l1 + l2)
            x_candidate = np.maximum(0.0, np.minimum(1.0, x * np.sqrt(np.maximum(0.0, -dc / (dv * lmid)))))
            if x_candidate.mean() - volfrac > 0:
                l1 = lmid
            else:
                l2 = lmid
        xnew = np.maximum(x - move, np.minimum(x + move, x_candidate))
        change = np.max(np.abs(xnew - x))
        x = xnew.copy()
        xPhys = x.copy()

        history['obj'].append(obj); history['vol'].append(xPhys.mean()); history['change'].append(change)

        if loop % display_interval == 0 or loop == 1 or change < tol:
            print(f"[Q4-{q4_mode}] Iter {loop}: Obj={obj:.6e}, mean_vol={xPhys.mean():.4f}, change={change:.4e}")

    density = xPhys.reshape((nelx, nely), order='F')
    density_plot = np.flipud(density.T)
    return density_plot, obj, loop, coords, edofMat, U, history

# ------------------------------
# T3 topology optimization (unchanged)
# ------------------------------
def topopt_T3(nelx=40, nely=20, volfrac=0.4, penal=3.0, rmin=1.5,
               maxiter=80, tol=1e-3, display_interval=10):
    D = D_matrix()
    coords, tri = mesh_T3_from_Q4(nelx, nely)

    nnodes = coords.shape[0]
    ndof = 2*nnodes
    nele = tri.shape[0]

    edofMat = np.zeros((nele,6), dtype=int)
    for el in range(nele):
        nodes = tri[el]
        edofMat[el,:] = np.array([2*nodes[0],2*nodes[0]+1, 2*nodes[1],2*nodes[1]+1, 2*nodes[2],2*nodes[2]+1])

    KE_list = []; area_list = []
    for el in range(nele):
        nodes = tri[el]
        coords_el = coords[nodes]
        ke, A = KE_tri(coords_el, D)
        KE_list.append(ke); area_list.append(A)
    KE_list = np.array(KE_list); area_list = np.array(area_list)

    F = np.zeros((ndof,1))
    mid_y = nely//2
    right_node = mid_y*(nelx+1) + nelx
    F[2*right_node+1,0] = -1.0

    left_node_bottom = 0
    left_node_mid = mid_y*(nelx+1)
    fixeddofs = np.array([2*left_node_bottom,2*left_node_bottom+1, 2*left_node_mid,2*left_node_mid+1])
    freedofs = np.setdiff1d(np.arange(ndof), fixeddofs)

    # Filter (index-based)
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
                H[i,j] = w; Hs[i] += w
    Hs += 1e-12

    x = volfrac * np.ones(nele); xPhys = x.copy()
    change = 1.0; loop = 0

    iK = np.kron(edofMat, np.ones((6,1))).flatten().astype(int)
    jK = np.kron(edofMat, np.ones((1,6))).flatten().astype(int)

    history = {'obj':[], 'vol':[], 'change':[]}

    while change > tol and loop < maxiter:
        loop += 1
        sK_vals = np.zeros(nele*36)
        for el in range(nele):
            s = Emin + xPhys[el]**penal * (E0 - Emin)
            sK_vals[el*36:(el+1)*36] = (s * KE_list[el]).reshape(36)
        K = sp.coo_matrix((sK_vals, (iK, jK)), shape=(ndof, ndof)).tocsc()

        Kff = K[freedofs[:,None], freedofs]
        Ff = F[freedofs,0]
        U = np.zeros((ndof,1))
        U_free = spla.spsolve(Kff, Ff)
        U[freedofs,0] = U_free

        ce = np.zeros(nele)
        for el in range(nele):
            ue = U[edofMat[el,:],0]
            ce[el] = ue.T @ KE_list[el] @ ue

        obj = ((Emin + xPhys**penal * (E0 - Emin)) * ce).sum()
        dc = -penal * xPhys**(penal-1) * (E0 - Emin) * ce

        dc = (H @ (x * dc)) / (Hs * np.maximum(1e-9, x))

        l1 = 0.0; l2 = 1e9; move = 0.2
        while (l2 - l1) / (l1 + l2 + 1e-9) > 1e-3:
            lmid = 0.5*(l1 + l2)
            B = np.maximum(0.0, np.minimum(1.0, x * np.sqrt(np.maximum(0.0, -dc/(lmid)))))
            if B.mean() - volfrac > 0:
                l1 = lmid
            else:
                l2 = lmid
        xnew = np.maximum(x - move, np.minimum(x + move, B))
        change = np.max(np.abs(xnew - x))
        x = xnew.copy(); xPhys = x.copy()

        history['obj'].append(obj); history['vol'].append(xPhys.mean()); history['change'].append(change)
        if loop % display_interval == 0 or loop == 1 or change < tol:
            print(f"[T3] Iter {loop}: Obj={obj:.6e}, mean_vol={xPhys.mean():.4f}, change={change:.4e}")

    # map triangles to quad grid (average two triangles per quad)
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
    return density_plot, obj, loop, coords, edofMat, U, history

# ------------------------------
# Run comparison
# ------------------------------
if __name__ == "__main__":
    nelx = 40; nely = 20
    volfrac = 0.7; penal = 3.0; rmin = 1.5
    maxiter = 80

    start = time.time()
    # Q4 reduced integration
    q4_red = topopt_Q4(nelx=nelx, nely=nely, volfrac=volfrac, penal=penal, rmin=rmin,
                       maxiter=maxiter, q4_mode='reduced', display_interval=20)
    q4_density_red, q4_obj_red, q4_iters_red, _, _, _, q4_hist_red = q4_red

    # Q4 Sigmund KE
    q4_sig = topopt_Q4(nelx=nelx, nely=nely, volfrac=volfrac, penal=penal, rmin=rmin,
                       maxiter=maxiter, q4_mode='sigmund', display_interval=20)
    q4_density_sig, q4_obj_sig, q4_iters_sig, _, _, _, q4_hist_sig = q4_sig

    # T3 (unchanged)
    t3_res = topopt_T3(nelx=nelx, nely=nely, volfrac=volfrac, penal=penal, rmin=rmin,
                       maxiter=maxiter, display_interval=20)
    t3_density, t3_obj, t3_iters, _, _, _, t3_hist = t3_res

    end = time.time()
    print("Total runtime (s):", end-start)

    # Plot side-by-side
    fig, axes = plt.subplots(1,3, figsize=(15,5))
    axes[0].imshow(q4_density_red, cmap=cm.gray, vmin=0, vmax=1); axes[0].axis('off')
    axes[0].set_title(f"Q4 reduced (1x1) Obj {q4_obj_red:.4f}, it {q4_iters_red}")
    axes[1].imshow(q4_density_sig, cmap=cm.gray, vmin=0, vmax=1); axes[1].axis('off')
    axes[1].set_title(f"Q4 Sigmund Obj {q4_obj_sig:.4f}, it {q4_iters_sig}")
    axes[2].imshow(t3_density, cmap=cm.gray, vmin=0, vmax=1); axes[2].axis('off')
    axes[2].set_title(f"T3 Obj {t3_obj:.4f}, it {t3_iters}")

    plt.tight_layout()
    plt.savefig("topopt_Q4_modes_vs_T3.png", dpi=200)
    plt.show()

    # also save individual images
    Image.fromarray((q4_density_red*255).astype(np.uint8)).convert("RGB").save("q4_reduced.png")
    Image.fromarray((q4_density_sig*255).astype(np.uint8)).convert("RGB").save("q4_sigmund.png")
    Image.fromarray((t3_density*255).astype(np.uint8)).convert("RGB").save("t3.png")

    print("Saved: q4_reduced.png, q4_sigmund.png, t3.png, topopt_Q4_modes_vs_T3.png")
