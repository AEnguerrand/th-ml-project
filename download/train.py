import zipfile
import os


def download_train():
    # SCRIPT NOT WORK ON WINDOWS
    # @todo path windows

    os.system('KAGGLE_USERNAME=aenguerrand KAGGLE_KEY=0f649bec92bbc12ba16e415a9e2c6b32 kaggle competitions download -c PLAsTiCC-2018 -p dataset -f training_set.csv')
    os.system('KAGGLE_USERNAME=aenguerrand KAGGLE_KEY=0f649bec92bbc12ba16e415a9e2c6b32 kaggle competitions download -c PLAsTiCC-2018 -p dataset -f training_set_metadata.csv')
    for file in os.listdir('dataset'):
        print("File", file)
        if len(file.split(".")) == 3:
          zip_ref = zipfile.ZipFile("dataset/" + file, 'r')
          zip_ref.extractall("dataset")
          zip_ref.close()
    print("File is unzip to csv. OK.")


if __name__ == '__main__':
    download_train()