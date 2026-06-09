# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Callable

class Laplace3DSolver:
    """
    3D Laplace solver on a cubic lattice with:
      6-point (face) stencil or 26-point stencil
      Jacobi, Gauss-Seidel, or SOR (with omega)
      Fixed Dirichlet boundary conditions by default (zero)
    """

    def __init__(
        self,
        size: int = 20,
        tolerance: float = 1e-6,
        max_iterations: int = 5000,
        stencil: str = "6-point",  # "6-point" or "26-point"
        method: str = "Jacobi",  # "Jacobi", "Gauss-Seidel", or "SOR"
        omega: float = 1.0,  # relaxation factor for SOR; 1.0 means no relaxation
    ):
        self.size = max(3, int(size))
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.stencil = stencil
        self.method = method
        self.omega = omega

        # Grids
        self.grid = np.zeros((self.size, self.size, self.size), dtype=float)
        self.next_grid = np.zeros_like(self.grid)

        # Boundary handling
        self._apply_boundary_conditions()

        # History for convergence diagnostics
        self.res_history = []
        self.iter_history = []

        # Precompute stencil offsets
        self.offsets = self._build_offsets(self.stencil)

    def _build_offsets(self, stencil: str):
        if stencil == "6-point":
            # 6 face-neighbors
            off = [(-1, 0, 0), (1, 0, 0),
                   (0, -1, 0), (0, 1, 0),
                   (0, 0, -1), (0, 0, 1)]
            return off
        elif stencil == "26-point":
            off = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if dx == dy == dz == 0:
                            continue
                        off.append((dx, dy, dz))
            return off
        else:
            raise ValueError("Unsupported stencil. Choose '6-point' or '26-point'.")

    def _apply_boundary_conditions(self):
        """
        Apply fixed Dirichlet boundaries. By default, zero on all faces.
        You can customize by passing a boundary_fn(i,j,k) or a boundary map.
        """
        s = self.size
        # Zero boundary by default
        self.grid[0, :, :] = 0.0
        self.grid[-1, :, :] = 0.0
        self.grid[:, 0, :] = 0.0
        self.grid[:, -1, :] = 0.0
        self.grid[:, :, 0] = 0.0
        self.grid[:, :, -1] = 0.0

    def set_boundary(self, boundary_fn: Optional[Callable[[int, int, int], float]] = None):
        """
        Optionally set nonzero/position-dependent boundary values via a function f(i,j,k) -> float.
        If boundary_fn is provided, it will be used to fill the six faces before solving.
        """
        if boundary_fn is None:
            self._apply_boundary_conditions()
            return

        s = self.size
        for i in range(s):
            for j in range(s):
                self.grid[0, i, j] = boundary_fn(0, i, j)
                self.grid[-1, i, j] = boundary_fn(s-1, i, j)
                self.grid[i, 0, j] = boundary_fn(i, 0, j)
                self.grid[i, -1, j] = boundary_fn(i, s-1, j)
                self.grid[i, j, 0] = boundary_fn(i, j, 0)
                self.grid[i, j, -1] = boundary_fn(i, j, s-1)

    def _update_jacobi(self):
        """
        Jacobi update: read from self.grid, write to self.next_grid.
        Interior points updated; boundaries copied from current grid.
        """
        s = self.size
        g = self.grid
        ng = self.next_grid

        # Copy boundary values
        ng[:, :, :] = g[:, :, :]

        if self.stencil == "6-point":
            for i in range(1, s-1):
                for j in range(1, s-1):
                    for k in range(1, s-1):
                        ng[i, j, k] = (
                            g[i-1, j, k] + g[i+1, j, k] +
                            g[i, j-1, k] + g[i, j+1, k] +
                            g[i, j, k-1] + g[i, j, k+1]
                        ) / 6.0
        else:
            # 26-point: average of all 26 neighbors
            for i in range(1, s-1):
                for j in range(1, s-1):
                    for k in range(1, s-1):
                        ssum = 0.0
                        for dx, dy, dz in self.offsets:
                            ssum += g[i+dx, j+dy, k+dz]
                        ng[i, j, k] = ssum / 26.0

    def _update_in_place(self):
        """
        Gauss-Seidel / SOR updates: in-place updates using latest values.
        """
        s = self.size
        g = self.grid

        if self.stencil == "6-point":
            for i in range(1, s-1):
                for j in range(1, s-1):
                    for k in range(1, s-1):
                        nb_sum = (
                            g[i-1, j, k] + g[i+1, j, k] +
                            g[i, j-1, k] + g[i, j+1, k] +
                            g[i, j, k-1] + g[i, j, k+1]
                        )
                        new_val = nb_sum / 6.0
                        if self.method == "SOR" and self.omega != 1.0:
                            new_val = (1.0 - self.omega) * g[i, j, k] + self.omega * new_val
                        g[i, j, k] = new_val
        else:
            for i in range(1, s-1):
                for j in range(1, s-1):
                    for k in range(1, s-1):
                        ssum = 0.0
                        for dx, dy, dz in self.offsets:
                            ssum += g[i+dx, j+dy, k+dz]
                        new_val = ssum / 26.0
                        if self.method == "SOR" and self.omega != 1.0:
                            new_val = (1.0 - self.omega) * g[i, j, k] + self.omega * new_val
                        g[i, j, k] = new_val

    def solve(self) -> int:
        """
        Run the selected iterative method until convergence or max_iterations.
        Returns the number of iterations performed.
        """
        self._apply_boundary_conditions()
        self.res_history = []
        self.iter_history = []

        for it in range(1, self.max_iterations + 1):
            if self.method == "Jacobi":
                self._update_jacobi()
                res = float(np.max(np.abs(self.next_grid - self.grid)))
                self.res_history.append(res)
                self.iter_history.append(it)
                self.grid, self.next_grid = self.next_grid, self.grid
            elif self.method in ("Gauss-Seidel", "SOR"):
                prev = self.grid.copy()
                self._update_in_place()
                res = float(np.max(np.abs(self.grid - prev)))
                self.res_history.append(res)
                self.iter_history.append(it)
            else:
                raise ValueError("Unknown method.")

            if res < self.tolerance:
                print(f"Converged in {it} iterations with residual {res:.2e}")
                return it

        print(f"Warning: Max iterations reached. Final residual: {res:.2e}")
        return self.max_iterations

    def plot_slice(self, axis: str = 'z', index: Optional[int] = None, title: str = "Solution Slice"):
        """
        Plot a 2D slice of the 3D solution.
        """
        if index is None:
            index = self.size // 2

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        if axis == 'x':
            data = self.grid[index, :, :]
            ax.set_xlabel('Y')
            ax.set_ylabel('Z')
        elif axis == 'y':
            data = self.grid[:, index, :]
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
        else:  # 'z'
            data = self.grid[:, :, index]
            ax.set_xlabel('X')
            ax.set_ylabel('Y')

        im = ax.imshow(data, cmap='viridis', origin='lower')
        plt.colorbar(im, ax=ax, label='Potential')
        ax.set_title(f"{title} ({axis}={index})")
        plt.tight_layout()
        plt.show()

    def plot_convergence(self):
        """
        Plot convergence history.
        """
        plt.figure(figsize=(8, 5))
        plt.semilogy(self.iter_history, self.res_history, 'b-', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Residual (max norm)')
        plt.title(f'Convergence History ({self.method}, {self.stencil})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def analytic_solution_example(i: int, j: int, k: int, size: int) -> float:
    """
    Example analytic solution for testing: 
    phi(x,y,z) = sin(pi*x/L) * sin(pi*y/L) * sin(pi*z/L)
    where x,y,z are normalized coordinates in [0,1]
    """
    x = i / (size - 1)
    y = j / (size - 1) 
    z = k / (size - 1)
    return np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)


def analytic_boundary_factory(size: int) -> Callable[[int, int, int], float]:
    def boundary(i, j, k):
        return analytic_solution_example(i, j, k, size)
    return boundary


def run_analytic_test(size: int = 21, stencil: str = "6-point", method: str = "Jacobi", 
                     tol: float = 1e-6, max_iter: int = 4000):
    """
    Validates solver against manufactured solution by setting boundary to the analytic phi
    and computing interior solution. After solving, compare interior values to analytic values.
    """
    solver = Laplace3DSolver(size=size, tolerance=tol, max_iterations=max_iter, 
                           stencil=stencil, method=method)
    solver.set_boundary(boundary_fn=analytic_boundary_factory(size))
    iters = solver.solve()
    
    # Compare interior points to analytic
    s = solver.size
    max_err = 0.0
    sum_err = 0.0
    count = 0
    for i in range(1, s-1):
        for j in range(1, s-1):
            for k in range(1, s-1):
                anal = analytic_solution_example(i, j, k, s)
                err = abs(solver.grid[i, j, k] - anal)
                max_err = max(max_err, err)
                sum_err += err
                count += 1

    mean_err = sum_err / count if count else 0.0
    print(f"Analytic test: iterations={iters}, max_err={max_err:.3e}, mean_err={mean_err:.3e}")

    # Optional: plot a central slice to visually compare
    solver.plot_slice(axis='z', index=size//2)
    solver.plot_convergence()

    return iters, max_err, mean_err


# Example usage
if __name__ == "__main__":
    # Quick demo: Jacobi with 6-point stencil
    size = 21
    solver = Laplace3DSolver(size=size, tolerance=1e-6, max_iterations=4000,
                           stencil="6-point", method="Jacobi")
    iters = solver.solve()
    print(f"Jacobi (6-point) converged in {iters} iterations.")
    solver.plot_slice(axis='z', index=size//2)
    solver.plot_convergence()

    # Optional: run analytic-validation test
    print("\n" + "="*50)
    print("Running analytic validation test:")
    run_analytic_test(size=21, stencil="6-point", method="Jacobi", tol=1e-6, max_iter=4000)

