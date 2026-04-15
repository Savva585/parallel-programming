import numpy as np
import argparse
import sys

def load_matrix(filename):
    with open(filename, 'r', encoding='utf-8-sig') as f:
        first = f.readline().strip()
        if not first:
            raise ValueError("Empty file")
        n = int(first)
        mat = []
        for _ in range(n):
            line = f.readline()
            if not line:
                raise ValueError("Not enough rows")
            parts = line.strip().split()
            if len(parts) != n:
                raise ValueError("Invalid row length")
            mat.append([float(x) for x in parts])
        return np.array(mat)

def main():
    parser = argparse.ArgumentParser(description='Verify matrix multiplication results.')
    parser.add_argument('fileA', help='First matrix file')
    parser.add_argument('fileB', help='Second matrix file')
    parser.add_argument('fileC', help='Result matrix file (from C++ program)')
    parser.add_argument('--tol', type=float, default=1e-5,
                        help='Tolerance for comparison (default: 1e-5)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed difference')
    args = parser.parse_args()

    try:
        A = load_matrix(args.fileA)
        B = load_matrix(args.fileB)
        C_cpp = load_matrix(args.fileC)

        C_ref = np.dot(A, B)

        if np.allclose(C_cpp, C_ref, rtol=args.tol, atol=args.tol):
            print("SUCCESS: matrices match within tolerance.")
        else:
            print("FAILURE: matrices differ.")
            diff = np.abs(C_cpp - C_ref)
            print("Max difference:", np.max(diff))
            if args.verbose:
                print("C++ result:\n", C_cpp)
                print("Reference result:\n", C_ref)
                print("Difference:\n", diff)
    except Exception as e:
        print("Error:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()