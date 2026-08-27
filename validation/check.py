import os

main_folder = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331"

for folder_name in os.listdir(main_folder):

    folder_path = os.path.join(main_folder, folder_name)

    if os.path.isdir(folder_path):

        print("\n================================")
        print("FOLDER:", folder_name)
        print("================================")

        for root, dirs, files in os.walk(folder_path):

            print("\nPath:", root)

            for file in files:
                print("   FILE:", file)