Context

Dataset is about cars from back in 85. It's raw and messy. We used a cleaned version from Kaggle
https://www.kaggle.com/fazilbtopal/auto85
called 'auto_clean.csv'

Content

This data set consists of three types of entities:
(a) the specification of an auto in terms of various characteristics, 
(b) its assigned insurance risk rating, 
(c) its normalized losses in use as compared to other cars. The second rating corresponds to the degree to which the auto is more risky than its price indicates. Cars are initially assigned a risk factor symbol associated with its price. Then, if it is more risky (or less), this symbol is adjusted by moving it up (or down) the scale. Actuarians call this process "symboling". A value of +3 indicates that the auto is risky, -3 that it is probably pretty safe. The third factor is the relative average loss payment per insured vehicle year. This value is normalized for all autos within a particular size classification (two-door small, station wagons, sports/specialty, etc…), and represents the average loss per car per year.

Note: Several of the attributes in the database could be used as a "class" attribute.

Number of Instances: 205

Number of Attributes: 26 total 
-- 15 continuous 
-- 1 integer 
-- 10 nominal

Attribute Information:

| Attribute | Attribute Range

symboling: -3, -2, -1, 0, 1, 2, 3
normalized-losses: continuous from 65 to 256
make: alfa-romero, audi, bmw, chevrolet, dodge, honda, isuzu, jaguar, mazda, mercedes-benz, mercury, mitsubishi, nissan, peugot, plymouth, porsche, renault, saab, subaru, toyota, volkswagen, volvo
fuel-type: diesel, gas
aspiration: std, turbo
num-of-doors: four, two
body-style: hardtop, wagon, sedan, hatchback, convertible
drive-wheels: 4wd, fwd, rwd
engine-location: front, rear
wheel-base: continuous from 86.6 120.9
length: continuous from 141.1 to 208.1
width: continuous from 60.3 to 72.3
height: continuous from 47.8 to 59.8
curb-weight: continuous from 1488 to 4066
engine-type: dohc, dohcv, l, ohc, ohcf, ohcv, rotor
num-of-cylinders: eight, five, four, six, three, twelve, two
engine-size: continuous from 61 to 326
fuel-system: 1bbl, 2bbl, 4bbl, idi, mfi, mpfi, spdi, spfi
bore: continuous from 2.54 to 3.94
stroke: continuous from 2.07 to 4.17
compression-ratio: continuous from 7 to 23
horsepower: continuous from 48 to 288
peak-rpm: continuous from 4150 to 6600
city-mpg: continuous from 13 to 49
highway-mpg: continuous from 16 to 54
price: continuous from 5118 to 45400.
Acknowledgements

"Automobile Data Set" from the following link: https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data.

Inspiration

I used these data for data cleaning/analysis purposes.

Version 2

****added a new data which is the clean version of our original data.