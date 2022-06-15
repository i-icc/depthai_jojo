import csv

if __name__ == "__main__":

    if input("初期化していいですか[y/n]") == "y":
        with open("./y_classified.csv", "w") as f:
            writer = csv.writer(f)
            # writer.writerow(["id", "people", "Giorno", "Jonathan"])
            writer.writerow(["id", "judge"])
            for i in range(13, 260):
                # writer.writerow([i, -1, -1, -1])
                if i < 38:
                    writer.writerow([i, 0])
                elif i < 72:
                    writer.writerow([i, 5])
                elif i < 79:
                    writer.writerow([i, 0])
                elif i < 107:
                    writer.writerow([i, 3])
                elif i < 114:
                    writer.writerow([i, 0])
                elif i < 140:
                    writer.writerow([i, 2])
                elif i < 178:
                    writer.writerow([i, 4])
                elif i < 239:
                    writer.writerow([i, 2])
                else:
                    writer.writerow([i, 0])
