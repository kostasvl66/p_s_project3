import subprocess
import re
import sys

# -----------------------------
# CONFIGURATION
# -----------------------------
Ns = [10000, 100000]
Ms = [4, 8, 16, 32, 64]
J = 2

EXECUTABLE = "make"
TARGET = "run-lab"
LOG_PH1 = "lab_benchmark_log_ph1.txt"
LOG_PH2 = "lab_benchmark_log_ph2.txt"
LOG_PH3 = "lab_benchmark_log_ph3.txt"
LOG  = "lab_benchmark_log.txt"

# -----------------------------
# FLOAT EXTRACTION REGEX
# -----------------------------
float_regex = re.compile(r"[-+]?\d*\.\d+")

# -----------------------------
# MAIN BENCHMARK
# -----------------------------
with open(LOG, "w") as log, open(LOG_PH1, "w") as log1, open(LOG_PH2, "w") as log2, open(LOG_PH3, "w") as log3:

    # ---------- HEADER ----------
    header = ["N\\M"]
    for M in Ms:
        header.append(f"{M}-serial")
        header.append(f"{M}-parallel")
        header.append(f"{M}-s/p")
        header.append(f"{M}-eff")

    log.write(" ".join(header) + "\n")

    header_ph = ["N\\M"]
    for M in Ms:
        header_ph.append(f"{M}-time")

    log1.write(" ".join(header_ph) + "\n")
    log2.write(" ".join(header_ph) + "\n")
    log3.write(" ".join(header_ph) + "\n")

    # ---------- ROWS ----------
    for N in Ns:
        row_values = [str(N)]
        row_values_ph1 = [str(N)]
        row_values_ph2 = [str(N)]
        row_values_ph3 = [str(N)]

        for M in Ms:
            ph1_sum = 0.0
            ph2_sum = 0.0
            ph3_sum = 0.0
            parallel_sum = 0.0
            serial_sum = 0.0

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

                if len(floats) != 5:
                    print("ERROR: Other than 5 floats found in output!")
                    print(result.stdout)
                    sys.exit(1)

                ph1_time = float(floats[0])
                ph2_time = float(floats[1])
                ph3_time = float(floats[2])
                parallel_time = float(floats[3])
                serial_time = float(floats[4])

                ph1_sum += ph1_time
                ph2_sum += ph2_time
                ph3_sum += ph3_time
                parallel_sum += parallel_time
                serial_sum += serial_time

            avg_ph1 = ph1_sum / J
            avg_ph2 = ph2_sum / J
            avg_ph3 = ph3_sum / J
            avg_parallel = parallel_sum / J
            avg_serial = serial_sum / J
            speedup = avg_serial / avg_parallel

            row_values.append(f"{avg_serial:.6f}")
            row_values.append(f"{avg_parallel:.6f}")
            row_values.append(f"{speedup:.6f}")
            row_values.append(f"{(speedup / M):.6f}")

            row_values_ph1.append(f"{avg_ph1:.6f}")
            row_values_ph2.append(f"{avg_ph2:.6f}")
            row_values_ph3.append(f"{avg_ph3:.6f}")

        log.write(" ".join(row_values) + "\n")
        log1.write(" ".join(row_values_ph1) + "\n")
        log2.write(" ".join(row_values_ph2) + "\n")
        log3.write(" ".join(row_values_ph3) + "\n")

print("Polynomial benchmark completed. Results written to: lab_benchmark_log*.txt")