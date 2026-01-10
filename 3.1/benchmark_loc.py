import subprocess
import re
import sys

# -----------------------------
# CONFIGURATION
# -----------------------------
Ns = [1000, 10000, 100000]
Ms = [1, 2, 4, 8, 15]
J = 4

EXECUTABLE = "make"
TARGET = "run-loc"
LOG_FILE = "loc_benchmark_log.txt"

# -----------------------------
# FLOAT EXTRACTION REGEX
# -----------------------------
float_regex = re.compile(r"[-+]?\d*\.\d+")

# -----------------------------
# MAIN BENCHMARK
# -----------------------------
with open(LOG_FILE, "w") as log:

    # ---------- HEADER ----------
    header = ["N\\M"]
    for M in Ms:
        header.append(f"{M}-serial")
        header.append(f"{M}-parallel")
        header.append(f"{M}-s/p")

    log.write(" ".join(header) + "\n")

    # ---------- ROWS ----------
    for N in Ns:
        row_values = [str(N)]

        for M in Ms:
            serial_sum = 0.0
            parallel_sum = 0.0

            for _ in range(J):
                result = subprocess.run(
                    [EXECUTABLE, TARGET, f"N={N}", f"P={M}"],
                    capture_output=True,
                    text=True
                )

                if result.returncode != 0:
                    print(f"Execution failed for N={N}, M={M}")
                    print(result.stderr)
                    sys.exit(1)

                floats = float_regex.findall(result.stdout)

                if len(floats) < 3:
                    print("ERROR: Less than 3 floats found in output!")
                    print(result.stdout)
                    sys.exit(1)

                # Take ONLY the LAST TWO floats
                serial_time = float(floats[-2])
                parallel_time = float(floats[-1])

                serial_sum += serial_time
                parallel_sum += parallel_time

            avg_serial = serial_sum / J
            avg_parallel = parallel_sum / J

            row_values.append(f"{avg_serial:.6f}")
            row_values.append(f"{avg_parallel:.6f}")
            row_values.append(f"{(avg_serial / avg_parallel):.6f}")

        log.write(" ".join(row_values) + "\n")

print("Polynomial benchmark completed. Results written to:", LOG_FILE)