import csv

if __name__ == "__main__":

    if input("初期化していいですか[y/n]") == "y":
        with open("./y_classified2.csv", "w") as f:
            writer = csv.writer(f)
            # writer.writerow(["id", "people", "Giorno", "Jonathan"])
            writer.writerow(["id", "judge"])
            for i in range(331):
                # writer.writerow([i, -1, -1, -1])
                if i < 81:
                    writer.writerow([i, 0])
                elif i < 146:
                    writer.writerow([i, 5])
                elif i < 154:
                    writer.writerow([i, 0])
                elif i < 257:
                    writer.writerow([i, 4])
                elif i < 306:
                    writer.writerow([i, 3])
                elif i < 331:
                    writer.writerow([i, 0])
                    
