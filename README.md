# Custom Paint by Number Generator
Turn a special photo into a paint by number!

## How to do it
- Verify you have [Python]<https://www.python.org/downloads/> 3.4 or higher installed
- Download the code
```
git clone repo
```
- Enter the Repo and set up a new a virtual python environment
```
cd repo
python3 -m venv venv
```
- Finally, make a paint by number
```
python3 generate.py --image ~/Downloads/your_image.jpeg --max-colors 15 --min-brush-size 500 --name-suffix my_image_test
```

## Results
In this example, you'll produce a couple files "paint_by_number_my_image_test.jpg", "preview_my_image_test.jpg", and "color_sheet_my_image_test.jpg"
