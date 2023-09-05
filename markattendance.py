import pandas as pd
import numpy as np

df = pd.read_csv("Attendance.csv")


names = df["Names"].unique()

try:
    name_df = pd.read_csv("Attendance_summary.csv")
    if "Names" not in name_df.columns:
        name_df = pd.DataFrame(columns=["Names", "Count", "Attendance"])
except FileNotFoundError:
    name_df = pd.DataFrame(columns=["Names", "Count", "Attendance"])


for name in names:
    if name in name_df["Names"].tolist():
        name_df.loc[name_df["Names"]==name, "Count"] += 1
    else:
        new_row = {"Names": name, "Count": 1, "Attendance": ""}
        name_df = pd.concat([name_df, pd.DataFrame(new_row, index=[0])], ignore_index=True)


name_df.to_csv("Attendance_summary.csv", index=False)


df.drop(df.index, inplace=True)
df.to_csv("Attendance.csv", index=False)

name_df["Attendance"] = np.where(name_df["Count"] >= 2, "Yes", np.nan)


name_df.to_csv("Attendance_summary.csv", index=False)