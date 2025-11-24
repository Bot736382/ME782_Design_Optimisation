import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import matplotlib.pyplot as plt

class TopologyOptimizer:
    def __init__(self, nelx, nely, volfrac, penal, rmin, ft):
        """
        Initialize the Topology Optimizer.
        """
        self.nelx = nelx
        self.nely = nely
        self.volfrac = volfrac
        self.penal = penal
        self.rmin = rmin
        self.ft = ft
        
        # Degrees of Freedom
        self.ndof = 2 * (self.nelx + 1) * (self.nely + 1)
        
        # Boundary Conditions Storage
        self.force_vector = np.zeros(self.ndof)
        self.fixed_dofs = []

        # Initial uniform density
        self.x = np.ones(nely * nelx) * volfrac
        self.xPhys = self.x.copy()
        
        # Element stiffness matrix (flat)
        E = 1.0
        nu = 0.3
        k = np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
        self.KE = E / (1 - nu**2) * np.array([
            [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
            [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
            [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
            [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
            [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
            [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
            [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
            [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
        ])

        # Pre-calculate filter
        self.H, self.Hs = self._prepare_filter()

    def _prepare_filter(self):
        nfilter = int(self.nelx * self.nely * ((2 * (np.ceil(self.rmin) - 1) + 1)**2))
        iH = np.zeros(nfilter, dtype=int)
        jH = np.zeros(nfilter, dtype=int)
        sH = np.zeros(nfilter)
        cc = 0
        for i in range(self.nelx):
            for j in range(self.nely):
                row = i * self.nely + j
                kk1 = int(np.maximum(i - (np.ceil(self.rmin) - 1), 0))
                kk2 = int(np.minimum(i + (np.ceil(self.rmin) - 1) + 1, self.nelx))
                ll1 = int(np.maximum(j - (np.ceil(self.rmin) - 1), 0))
                ll2 = int(np.minimum(j + (np.ceil(self.rmin) - 1) + 1, self.nely))
                for k in range(kk1, kk2):
                    for l in range(ll1, ll2):
                        col = k * self.nely + l
                        fac = self.rmin - np.sqrt(((i-k)*(i-k) + (j-l)*(j-l)))
                        iH[cc] = row
                        jH[cc] = col
                        sH[cc] = np.maximum(0.0, fac)
                        cc = cc + 1
        
        H = sp.coo_matrix((sH, (iH, jH)), shape=(self.nelx*self.nely, self.nelx*self.nely)).tocsc()
        Hs = H.sum(1)
        return H, Hs

    def set_load(self, x, y, fx, fy):
        node_idx = x * (self.nely + 1) + y
        if fx != 0: self.force_vector[2 * node_idx] += fx
        if fy != 0: self.force_vector[2 * node_idx + 1] += fy

    def set_fixed_support(self, x_min, x_max, y_min, y_max, fix_x=True, fix_y=True):
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                node_idx = x * (self.nely + 1) + y
                if fix_x: self.fixed_dofs.append(2 * node_idx)
                if fix_y: self.fixed_dofs.append(2 * node_idx + 1)
        self.fixed_dofs = list(set(self.fixed_dofs))

    def FE_analysis(self):
        # 1. Compute EDof matrix (Map element -> global DOFs)
        edofMat = np.zeros((self.nelx * self.nely, 8), dtype=int)
        for elx in range(self.nelx):
            for ely in range(self.nely):
                el = elx * self.nely + ely
                n1 = (self.nely + 1) * elx + ely
                n2 = (self.nely + 1) * (elx + 1) + ely
                edofMat[el, :] = np.array([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3, 2*n2, 2*n2+1, 2*n1, 2*n1+1])
        
        # 2. Construct Sparse Global Stiffness Matrix K
        iK = np.repeat(edofMat, 8, axis=1).flatten()
        jK = np.tile(edofMat, (1, 8)).flatten()
        sK = ((self.KE.flatten()[np.newaxis]).T * (self.xPhys**self.penal)).flatten(order='F')
        
        K = sp.coo_matrix((sK, (iK, jK)), shape=(self.ndof, self.ndof)).tocsc()

        # 3. Solve
        F = self.force_vector
        fixed = np.array(self.fixed_dofs)
        free = np.setdiff1d(np.arange(self.ndof), fixed)
        
        u = np.zeros(self.ndof)
        u[free] = scipy.sparse.linalg.spsolve(K[free, :][:, free], F[free])
        return u, edofMat

    def optimize(self, max_loop=50):
        if len(self.fixed_dofs) == 0:
            raise ValueError("No supports defined!")

        print(f"Optimizing {self.nelx}x{self.nely} grid...")
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 1. Plot Material (Gray scale)
        # Note: origin='lower' puts y=0 at bottom.
        im = ax.imshow(-self.xPhys.reshape((self.nelx, self.nely)).T, 
                       cmap='gray', interpolation='none', 
                       vmin=-1, vmax=0, origin='lower')
        
        # 2. Visualize Boundary Conditions (Supports)
        # Extract x, y coordinates from fixed_dofs
        fixed_x_coords = []
        fixed_y_coords = []
        for dof in self.fixed_dofs:
            node_idx = dof // 2
            x = node_idx // (self.nely + 1)
            y = node_idx % (self.nely + 1)
            fixed_x_coords.append(x - 0.5) # Offset slightly to align with grid intersections
            fixed_y_coords.append(y - 0.5)
        
        # Plot supports as Red Triangles
        ax.scatter(fixed_x_coords, fixed_y_coords, c='red', marker='^', s=30, label='Fixed Support', zorder=5)

        # 3. Visualize Forces (Loads)
        # Find non-zero forces
        for i in range(0, self.ndof, 2):
            fx = self.force_vector[i]
            fy = self.force_vector[i+1]
            if fx != 0 or fy != 0:
                node_idx = i // 2
                x = node_idx // (self.nely + 1)
                y = node_idx % (self.nely + 1)
                
                # Scale arrow for visibility
                scale = 10.0 
                ax.arrow(x - 0.5, y - 0.5, fx * scale, fy * scale, 
                         head_width=2, head_length=2, fc='blue', ec='blue', zorder=10)
                
                # Label the force value
                label_text = f"Fx:{fx:.1f}\nFy:{fy:.1f}" if fx!=0 and fy!=0 else (f"Fx:{fx:.1f}" if fx!=0 else f"Fy:{fy:.1f}")
                ax.text(x + fx*scale, y + fy*scale, label_text, color='blue', fontweight='bold')

        ax.set_title("Initial Design")
        ax.set_xlim(-5, self.nelx + 5)
        ax.set_ylim(-5, self.nely + 5)
        ax.legend(loc='upper right')
        fig.canvas.draw()
        plt.pause(0.1)

        # Main Loop
        save_iterations = [0, 12, 24, 36, 49]
        for loop in range(max_loop):
            u, edofMat = self.FE_analysis()
            
            # Sensitivity Analysis
            ue = u[edofMat]
            ce = np.sum(np.dot(ue, self.KE) * ue, axis=1)
            c = np.sum((self.xPhys**self.penal) * ce)
            dc = -self.penal * (self.xPhys**(self.penal - 1)) * ce
            
            # Filtering
            dc[:] = np.asarray((self.H * (self.x * dc))[np.newaxis].T / self.Hs)[:, 0] / np.maximum(0.001, self.x)
            
            # Update
            self.x = self._optimality_criteria_update(dc)
            self.xPhys = self.x
            
            # Plot
            print(f"It.: {loop:4d} | Compliance: {c:10.4f} | Vol: {np.mean(self.xPhys):.3f}")
            
            im.set_data(-self.xPhys.reshape((self.nelx, self.nely)).T)
            ax.set_title(f"Iter {loop}: Compliance = {c:.2f}")
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)
            
            # Save snapshots at specified iterations
            if loop in save_iterations:
                fig_snap, ax_snap = plt.subplots(figsize=(10, 10))
                ax_snap.imshow(-self.xPhys.reshape((self.nelx, self.nely)).T, 
                               cmap='gray', interpolation='none', 
                               vmin=-1, vmax=0, origin='lower')
                
                # Add supports (red triangles)
                fixed_x_coords = []
                fixed_y_coords = []
                for dof in self.fixed_dofs:
                    node_idx = dof // 2
                    x_coord = node_idx // (self.nely + 1)
                    y_coord = node_idx % (self.nely + 1)
                    fixed_x_coords.append(x_coord - 0.5)
                    fixed_y_coords.append(y_coord - 0.5)
                
                ax_snap.scatter(fixed_x_coords, fixed_y_coords, c='red', marker='^', s=50, label='Fixed Support', zorder=5)
                
                # Add forces (blue arrows)
                scale = 10.0
                for i in range(0, self.ndof, 2):
                    fx = self.force_vector[i]
                    fy = self.force_vector[i+1]
                    if fx != 0 or fy != 0:
                        node_idx = i // 2
                        x_coord = node_idx // (self.nely + 1)
                        y_coord = node_idx % (self.nely + 1)
                        ax_snap.arrow(x_coord - 0.5, y_coord - 0.5, fx * scale, fy * scale, 
                                     head_width=2, head_length=2, fc='blue', ec='blue', zorder=10)
                        label_text = f"Fx:{fx:.1f}\nFy:{fy:.1f}" if fx!=0 and fy!=0 else (f"Fx:{fx:.1f}" if fx!=0 else f"Fy:{fy:.1f}")
                        ax_snap.text(x_coord + fx*scale, y_coord + fy*scale, label_text, color='blue', fontweight='bold', fontsize=8)
                
                ax_snap.set_title(f"Topology at Iteration {loop}", fontsize=14)
                ax_snap.set_xlabel("X")
                ax_snap.set_ylabel("Y")
                ax_snap.legend(loc='upper right')
                filename = f"topology_iter_3_{loop}.png"
                fig_snap.savefig(filename, dpi=150, bbox_inches='tight')
                print(f"Saved {filename}")
                plt.close(fig_snap)

        plt.ioff()
        plt.show()

    def _optimality_criteria_update(self, dc):
        l1, l2, move = 0, 100000, 0.2
        xnew = np.zeros(self.nelx * self.nely)
        
        while (l2 - l1) / (l1 + l2) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            xnew = np.maximum(0.001, np.maximum(self.x - move, np.minimum(1.0, np.minimum(self.x + move, self.x * np.sqrt(-dc / lmid)))))
            
            if np.sum(xnew) - self.volfrac * self.nelx * self.nely > 0:
                l1 = lmid
            else:
                l2 = lmid
        return xnew

if __name__ == "__main__":
    # Create 60x60 square
    nelx, nely = 100,100
    opt = TopologyOptimizer(nelx=100, nely=100, volfrac=0.3, penal=3.0, rmin=1.5, ft=0)
    
    # --- HOW TO ADD MULTIPLE SUPPORTS ---
    # You can call set_fixed_support() as many times as you like.
    
    # 1. Fix the entire bottom edge (Your request)
    # x goes from 0 to 60, y is fixed at 0
    opt.set_fixed_support(x_min=0, x_max=nelx, y_min=0, y_max=0)
    
    # Example of adding a SECOND support (e.g., pinning the top-left corner)
    # opt.set_fixed_support(x_min=0, x_max=0, y_min=60, y_max=60) 

    # --- HOW TO ADD MULTIPLE LOADS ---
    # You can call set_load() as many times as you like. Forces at the same node will add up.
    
    # 1. Main Load: Downward force at Top-Middle (Your request)
    opt.set_load(x=0, y=100, fx=2.0, fy=-1.0)
    
    # 2. Secondary Load: Side push at the Top-Right (Demonstration)
    # This simulates a complex loading scenario
    # opt.set_load(x=30, y=30, fx=-1, fy=1) 

    # Run optimization
    opt.optimize(max_loop=500)