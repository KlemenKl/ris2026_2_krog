import csv
import shutil
import os


def parse_into_folders(file_name, output_folder, input_folder):
    with open(file_name, newline='') as csvfile:
        # print("There while: ")
        column_names = []
        spamreader = csv.reader(csvfile, delimiter=',')
        all_rows = 0
        for row in spamreader:
            all_rows += 1
            # print("There: "+row[0])
            if column_names == []:
                column_names = row
                continue
            directory = output_folder+"/"+row[1]+"/"
            if not os.path.exists(directory):
                os.makedirs(directory)
            if os.path.exists(input_folder+"/"+row[0]):
                # print("Copying: "+row[0]+" to "+directory)
                shutil.copy(input_folder+"/"+row[0], output_folder+"/"+row[1]+"/"+row[0])
            else:
                print("File not found: "+input_folder+"/"+row[0])
        print(f"Total rows processed: {all_rows}")

def read_csv(file_name):
    data = {}
    classes = []
    column_names = []
    with open(file_name, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            if column_names == []:
                column_names = row
                continue
            if row[1] not in classes:
                classes.append(row[1])
            data[row[0]] = row[1]

    return data, classes


parse_into_folders(file_name='train.csv', output_folder = "data/train/", input_folder= "ucni_set")

# if __name__ == "__main__":
#     parse_into_folders(file_name='ucni_set.csv', output_folder = "data/train/", input_folder= "ucni_set")
