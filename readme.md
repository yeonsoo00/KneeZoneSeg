# Knee Growth Plate Zone Segmentation

Staining signals : Mineral(1), AC(2), Calcein(3), TRAP(4), DAPI(5), AP(6), EdU(7), CFO(8), SFO(9)
(46 = 4 + 6 + 4 + 4 + 4 + 8 + 6 + 6 + 4)
Number of masks per signal:
Mineral(1) : 1a, 1b, 1c, 1d (counts = 4)
AC(2) : 2a, 2b, 2c, 2d, 2e, 2f (counts = 6)
Calcein(3) : 3a, 3b, 3c, 3d (counts = 4)
TRAP(4) : 4a, 4b, 4c, 4d (counts = 4)
DAPI(5) : 5a, 5b, 5c, 5d (counts = 4)
AP(6) : 6a, 6b, 6c, 6d, 6e, 6f + 6g, 6h (counts = 8)
EdU(7): 7a, 7b, 7c, 7d + 7e, 7f (counts = 6)
CFO(8) : 8a, 8b, 8c, 8d + 8e, 8f (counts = 6)
SFO(9) : 9a, 9b, 9c, 9d (counts = 4)

To get 9A, 9B we use SFO and CFO images. And 8E, 8F are defined from CFO and AP. 6G and 8H are defined by AP and TRAP. The rest of lines/zones are defined by the corresponding signals.
(a,b) and (c,d) are representing the high intensity line/zone or the low intensity line/zone.
Here, (a,b), (c,d), (e,f) pairs represent the same line but upper or lower zones. Since we want to make distance or area gap metrics across the signals, we made upper and lower representations. 

![Alt text]([path/to/image.png](https://github.com/yeonsoo00/KneeZoneSeg/blob/main/src/CCC_K10_M4_L1/mineral.png))


| Mineral | Zone1 | Zone2 |
| :---: | :---: | :---: |
| ![alt text 1]((https://github.com/yeonsoo00/KneeZoneSeg/blob/main/src/CCC_K10_M4_L1/mineral.png)) | ![alt text 2]([image2_url](https://github.com/yeonsoo00/KneeZoneSeg/blob/main/src/overlay/1b_overlay.png)) | ![alt tmext 3](https://github.com/yeonsoo00/KneeZoneSeg/blob/main/src/overlay/1d_overlay.png)|

