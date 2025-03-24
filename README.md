# Color Detection Tool Using Machine Learning

### By Suinan Xiao

Here I present a machine learning-based color detection tool that accurately identifies photographed color samples. The model predicts accurate RGB values given color samples despite varying lighting conditions. The model achieved results with an overall mean absolute error (MAE) of 5.50, a R² of 0.854, and a mean absolute percentage error (MAPE) of 3.02%. This tool is a potential solution for color matching tasks and eliminates the need for physical color samples.

GitHub Repository: [https://github.com/SUINANTHEBUG/Color_Recognition](https://github.com/SUINANTHEBUG/Color_Recognition)

## 1. Introduction

Many everyday tasks depend on matching specific colors accurately. Traditionally, verifying exact shades and hues has required physically carrying samples to retail locations such as hardware stores or paint suppliers to match them against extensive color swatches. This process is time-consuming and inconvenient. With the proliferation of high-quality camera sensors and advanced machine learning methods, an automated color detection solution can significantly streamline the process.

The objective of this project is to develop a tool that can correctly identify and calibrate a photographed color sample. By utilizing a standardized color card—consisting of three distinct reference shapes in red, green, and blue—the system is able to automatically detect and measure each shape's observed RGB values. Because the true RGB values for each reference color are known, the tool can then adjust the observed sample color to compensate for lighting and other environmental factors, returning a more precisely calibrated RGB value.

## 2. Data Acquisition and Preparation

In order to train and evaluate the calibration pipeline, images were gathered by members of the class, each containing both the standardized color card (red circle, green triangle, and blue pentagon) and various color samples with known true RGB values. These samples originated from commercial color samples sold in Home Depot. The true R, G, and B values for each color sample are available online.

![Standardized color card and color sample](https://github.com/SUINANTHEBUG/Color_Recognition/raw/main/images/image1.png)
*Figure 1: Standardized color card and color sample*

Once an image is captured, the system first runs it through a YOLO-based visual recognition module that has been specifically adapted for the standardized color card. In this module, the YOLO network—integrated through Pang Liu's code—locates predefined regions on the card. The network is trained to detect several key shapes, namely the red_circle, green_triangle, blue_pentagon, and a black_box region. For each detected region, the algorithm extracts the corresponding bounding box and calculates the average RGB values within that area. These extracted values serve two purposes: the reference regions (red, green, and blue) provide the true color values for calibration, and the observed sample region's color is adjusted based on the discrepancies between the observed and known reference values. This process allows the system to robustly correct for variations in lighting or sensor bias by comparing the extracted sample color to the calibrated benchmark provided by the color card. For more technical details, please refer to the implementation in the [Color_Calibration](https://github.com/jeffliulab/Color_Calibration) repository.

![Visual recognition of the sample color and reference colors](https://github.com/SUINANTHEBUG/Color_Recognition/raw/main/images/image2.png)
*Figure 2: Visual recognition of the sample color and reference colors. (Pang Liu)*

After detection, a table of color data was created, stored in CSV format. Each entry included 4 parts:

1. True R, True G, True B: the known RGB of the sample color.
2. Observed R, Observed G, Observed B: the directly measured RGB of the sample in the photo before calibration.
3. Circle R, Circle G, Circle B; Triangle R, Triangle G, Triangle B; Pentagon R, Pentagon G, Pentagon B: the measured RGB values for the reference colors.
4. camera: categorical data of different devices or cameras used by different people.

Each RGB value is an integer between 0–255; the higher the number, the greater the intensity of that color. Because multiple devices, environmental conditions, and angles were involved, the final dataset captured a variety of scenarios, thereby reducing bias (or "batch effects") during model training.

Because the data was collected by different people, there appeared to be a large discrepancy across reference colors—likely due to mislabeling (e.g., red labeled as blue and vice versa) and poor lighting. To address this, the dataset was cleaned.

![Color distribution of reference colors](https://github.com/SUINANTHEBUG/Color_Recognition/raw/main/images/image3.png)
*Figure 3: Color distribution of reference colors across a portion of all the samples collected.*

After removing mislabeled samples, outliers were identified and removed using the Interquartile Range (IQR) rule applied separately to each RGB channel of the reference colors. The reference colors were then transformed into a perceptual color space using a pseudo-hue metric, calculated as the angle in the RGB color cube to approximate human color perception. Each RGB triplet was normalized and converted to a pseudo-hue by projecting the color vector onto a 2D plane and calculating its angle relative to a centered red axis. Figure 4 shows well-separated peaks for red, green, and blue clusters. Red is now centered around ~30°, green around ~160°, and blue around ~260°, demonstrating better consistency across labeled categories.

![Filtered color distribution](https://github.com/SUINANTHEBUG/Color_Recognition/raw/main/images/image4.png)
*Figure 4: Filtered color distribution after IQR-based outlier removal and hue correction.*

## 3. Color Calibration Model

For the color calibration task, I implemented a **gradient boosted decision trees via the XGBoost framework**. The feature set contains 12 dimensions: the observed RGB values of the sample color and the observed RGB values of the three reference shapes (red circle, green triangle, and blue pentagon). These features encapsulate both the target color information and the environmental conditions represented by the reference colors.

I trained separate XGBoost regression models for each RGB channel, enabling independent optimization and more nuanced performance analysis per color component. The dataset was stratified by camera type during the 90:10 train-test split to ensure generalization across different devices. I implemented a **5-fold cross-validation**, maintaining camera distribution across folds, with model predictions rounded to valid RGB integer values (0-255). The XGBoost models used a squared error regression objective function with default parameters (L = ∑(yᵢ - ŷᵢ)²).

I computed multiple evaluation metrics: Mean Absolute Error (MAE) for average error magnitude, Root Mean Squared Error (RMSE) to penalize larger errors, Coefficient of Determination (R²) for explained variance, Mean Absolute Percentage Error (MAPE) for relative error, and Delta E (ΔE) for perceptually relevant color difference in CIE Lab space. After cross-validation, I compared two approaches: using the best-performing models from cross-validation for each RGB channel, and training new models on the entire training dataset. Both were evaluated on the held-out test set, with the better-performing strategy (based on overall MAE) selected for the final system.

The final prediction system consists of three independent XGBoost models, one for each RGB channel. When making predictions, the observed sample color and reference values are processed through each model, producing channel predictions that combine to form the calibrated RGB color.

## 4. Results

The color calibration model achieved promising results across the evaluation metrics. The full training set models outperformed the cross-validation models, with an overall MAE of 5.50 compared to 6.15, demonstrating the benefit of leveraging all available training data. Over half of the samples were predicted without error, and 76.19% within acceptable margin of error (MAE = 5)

When examining channel-specific performance, the model showed varying degrees of accuracy:

![Channel performance metrics](https://github.com/SUINANTHEBUG/Color_Recognition/raw/main/images/image5.png)

The perceptual color difference in CIE Lab space (ΔE) yielded a mean of 5.10 and a median of 0.63, indicating that most predictions were close to the true colors despite some outliers with larger errors. Direct observation of the difference between the true and predicted color (Figure 5) can more directly prove this point.

![True and predicted color comparison](https://github.com/SUINANTHEBUG/Color_Recognition/raw/main/images/image6.png)
*Figure 5: A portion of the true and predicted color comparison, sampled in proportion to the overall accuracy.*

MAE threshold analysis (Figure 6) revealed that 76.19% of predictions fell within 5 units of MAE on average across channels, with G (82.86%) and B (84.29%) channels significantly outperforming the R channel (61.43%). Similarly, while 93.33% of predictions fell within 20 units of MAE on average, the R channel (87.14%) lagged behind G (97.14%) and B (95.71%).

![Percentage of the Prediction following MAE Threshold](https://github.com/SUINANTHEBUG/Color_Recognition/raw/main/images/image7.png)
*Figure 6: Percentage of the Prediction following MAE Threshold*

For all three models, the observed value of the respective channel was the most important feature, with varying degrees of contribution from other features. The R channel model showed a more distributed importance pattern, with Observed R at ~16% importance, followed by Observed G at ~14%. In contrast, the G and B channel models showed much higher importance for their respective observed values (G at ~43% and B at ~51%).

![Feature importance for each RGB channel model](https://github.com/SUINANTHEBUG/Color_Recognition/raw/main/images/image8.png)
*Figure 7: Feature importance for each RGB channel model.*

The R channel's underperformance relative to G and B can be due to several factors. First, red wavelengths tend to be more susceptible to variations in ambient lighting conditions, particularly under indoor lighting which often has a yellowish tint affecting red perception. Second, most digital camera sensors have different sensitivities across wavelengths, with typically lower quantum efficiency in the red spectrum. Third, as evidenced by the feature importance distribution, the red channel prediction relies on a more complex combination of features, making it more challenging to model accurately.

Despite these challenges, the overall system achieved high accuracy with a combined R² of 0.854 and MAPE of 3.02% across all channels, making it effective for practical color detection and calibration applications.

## 5. Discussion

The color calibration model shows promise, with several avenues for improvement. The red channel's underperformance (MAE = 10.04 versus ~3.2 for green and blue) could be addressed through specialized model architectures, alternative color spaces, or additional features that better account for red wavelength's sensitivity to lighting variations. Exploring deep learning approaches or augmenting the training data with simulated lighting conditions could further enhance the model's robustness to environmental variables.

In a commercial setting, the system would benefit greatly from controlled data acquisition. A practical application would include guidance for proper photo taking, quality checks, and better quality reference cards. These implementations would yield performance metrics much better than our current results (R² = 0.854, MAPE = 3.02%).

### Reference:

Liu, P. (2025). Color_Calibration [Computer software]. GitHub. [https://github.com/jeffliulab/Color_Calibration](https://github.com/jeffliulab/Color_Calibration)
