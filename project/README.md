Clone the repository/download it to local folder

Download the dataset from google drive

```
https://drive.google.com/drive/folders/1hm1a16BsNe9ohK6OdvwWqhKz61Cpsr8r?usp=sharing
```

Download pre-trained model from the following url, {too large for github}

```
https://drive.google.com/file/d/1vvQUINJMMoSofUwq1RhIGgKJAUKbvWC9/view?usp=sharing
```

Move into project folder

Run requirements.txt via bash file to ensure dependencies are installed

```
Dependencies:
pillow
numpy
scipy
tensorflow
keras
h5py
pandas
```
```
run bash script

sudo bash install.sh
```

Run Main.py to run project. Main file expects the following arguements

- train/test variable: 1 indicates train from scratch, 0 indicates out of box prediction
- Path to Dataset
- Path to Pretrained Model

```
Project/Code:> python3 Main.py 0 'path_to_dataset/2013/' 'path_to_model/' 

Do not forget the slashes at the end
```

Output

![predicted](https://user-images.githubusercontent.com/29556523/33593643-455070a6-d95e-11e7-9894-2d663a3e70f0.jpg)

