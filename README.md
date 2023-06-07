# Handheld SR Satellites

![image](https://github.com/Jamy-L/handheld-sr-satellite/assets/46826148/dedf2a10-863f-4c68-8dfd-4abbc0b0f423)

**Good News: Our Paper Accepted at CVPR 2023!**

We are pleased to introduce the Handheld SR Satellites project, which applies the Handheld Burst Super Resolution algorithm to satellite imagery. Our paper, authored by Jamy Lafenetre, Ngoc Long Nguyen, Gabriele Facciolo and Thomas Eboli, has been accepted at CVPR 2023! You can find the paper [here]([link_to_paper](https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/papers/Lafenetre_Handheld_Burst_Super-Resolution_Meets_Multi-Exposure_Satellite_Imagery_CVPRW_2023_paper.pdf)).

## Requirements

To run the code, make sure you have the following dependencies installed:

- numpy==1.23.4
- numba==0.56.4
- torch==1.13.1
- scipy==1.9.1
- opencv-python==4.6.0.66
- matplotlib==3.5.2
- scikit-image==0.19.3

You can also find the complete list of requirements in the requirements.txt file.

## Usage

To run the code, follow these steps:

1. Clone this repository:

   ```
   git clone https://github.com/username/handheld-sr-satellite.git
   ```

2. Install the required dependencies using pip:

   ```
   pip install -r requirements.txt
   ```

3. Open the `example.py` file and modify the parameters according to your needs:

   ```
   params = {
       "scale": 2,
       "base detail": True,  # Whether to use the base detail decomposition or not
       "alignment": "Fnet",  # 'Fnet', 'ICA', 'patch ICA', the alignment method
   }
   ```
   Note that the global ICA require to install the icaflow package available [here](link)

4. Run the `example.py` script:

   ```
   python example.py
   ```

## Example

Here are some examples of the Handheld SR Satellites algorithm applied to satellite imagery:

![Example_1](https://github.com/Jamy-L/handheld-sr-satellite/assets/46826148/05cdb6e0-0525-4cb4-8914-255e0a0cfffc)

![Example_2](https://github.com/Jamy-L/handheld-sr-satellite/assets/46826148/1f294439-8a59-4bb0-81d8-2ecf57a9939e)

## Evaluation Data

Synthetic evaluation data for the Handheld SR Satellites project are available in the latest release. You can access the release [here](link_to_release) to explore and evaluate our algorithm.

## License

This project is licensed under the MIT License. Feel free to utilize the algorithm and contribute to its development.
