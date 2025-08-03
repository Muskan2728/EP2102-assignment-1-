import csv
import time
from datetime import datetime

name = input("Enter your name: ")
filename = f"{name}_log.csv"

with open(filename, mode='w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(["Name", "Timestamp", "Time Delay (s)", "Number Entered"])

    t0 = time.time()

    while True:
        x = input("Enter a number between 0-9 (or type 'quit' to exit): ")

        if x.lower() == "quit":
            print("Exiting program.")
            break

        if not x.isdigit() or not (0 <= int(x) <= 9):
            print("Invalid input. Please enter a digit between 0-9.")
            continue

        t1 = time.time()
        dt = t1 - t0
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        wr.writerow([name, ts, f"{dt:.2f}", x])
        print(f"Recorded: {x} at {ts} (Delay: {dt:.2f}s)")
        t0 = t1
