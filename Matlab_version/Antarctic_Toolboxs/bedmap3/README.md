# Bedmap3 tools for MATLAB
This repo contains MATLAB functions to easily work with [Bedmap3](https://doi.org/10.1038/s41597-025-04672-y) Antarctic Ice Sheet topography data. 

# Requirements 
To use these functions you'll need two things: 

1. The `bedmap3.nc` data file, which can be found [here](https://doi.org/10.5285/2d0e4791-8e20-46a3-80e4-f5f6716025d2), and 
2. [Antarctic Mapping Tools for MATLAB](https://github.com/chadagreene/Antarctic-Mapping-Tools).

# Documentation 
For simple text documentation, type `help` followed by the name of the function you want to learn about. For example:
 
```matlab 
help bedmap3_data
```
Unfortunately, I haven't written exhaustive documentation for these functions, and I'm not sure if or when I'll get around to it. The functions in this repository are modeled after my [BedMachine functions](https://github.com/chadagreene/BedMachine), however, so most of the BedMachine examples will work with these Bedmap3 functions.

# Function List 
* **`bedmap3_data`** loads Bedmap3 data into MATLAB. 
* **`bedmap3_interp`** interpolates Bedmap3 data to any geographic or polar stereographic coordinates. 
* **`bedmap3`** creates simple plots of Bedmap3 data. 
* **`bedmap3_profile`** creates a 2D profile along any given line. 

# Citation 
If you use Bedmap3 data, please cite the Pritchard paper listed below. And if this function is useful for you, please do me a kindness and cite my Antarctic Mapping Tools paper, too. 

* Pritchard, H.D., Fretwell, P.T., Fremand, A.C. et al. Bedmap3 updated ice bed, surface and thickness gridded datasets for Antarctica. *Sci Data* 12, 414 (2025). [https://doi.org/10.1038/s41597-025-04672-y](https://doi.org/10.1038/s41597-025-04672-y)
* Greene, C. A., Gwyther, D. E., & Blankenship, D. D. Antarctic Mapping Tools for Matlab. *Computers & Geosciences.* 104 (2017) pp.151-157. [http://dx.doi.org/10.1016/j.cageo.2016.08.003](https://doi.org/10.1038/s41597-025-04672-y)