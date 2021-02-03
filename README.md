# Custom Paint by Number Generator
Turn a special photo into a paint by number!

## Example Input
![alt text](https://github.com/jnatale11/CustomPaintByNumberGenerator/blob/main/examples/sm_beach.jpg?raw=true) 

## Example Output
![alt text](https://github.com/jnatale11/CustomPaintByNumberGenerator/blob/main/examples/paint_by_number_sm_beach.jpg?raw=true) ![alt text](https://github.com/jnatale11/CustomPaintByNumberGenerator/blob/main/examples/preview_sm_beach.jpg?raw=true) ![alt text](https://github.com/jnatale11/CustomPaintByNumberGenerator/blob/main/examples/color_sheet_sm_beach.jpg?raw=true)

## How to run it (On Mac or Linux)
- Verify you have Python 3.4 or higher installed <https://www.python.org/downloads/>
- Download the code
```
git clone https://github.com/jnatale11/CustomPaintByNumberGenerator.git
```
- Enter the Repo and set up a new a virtual python environment
```
cd CustomPaintByNumberGenerator
python3 -m venv my_venv
source my_venv/bin/activate
pip install -r requirements.txt
```
- Finally, make a paint by number
```
python3 generate.py --image ~/Downloads/your_image.jpeg --max-colors 15 --min-brush-size 500 --name-suffix my_image_test
```

## Results
As the above example shows, you'll produce a couple files- "paint_by_number_my_image_test.jpg", "preview_my_image_test.jpg", and "color_sheet_my_image_test.jpg". Play around with the "max-colors" and "min-brush-size" inputs as needed to get the perfect paint by number. Enjoy :)
