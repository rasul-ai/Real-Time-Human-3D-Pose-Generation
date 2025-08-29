import numpy as np
import json

keypoints_2d = np.array([[
    [3.73933197e+02, 4.51167908e+02, 3.17785814e-02],
    [2.17838394e+02, 4.67677948e+02, 2.96865404e-02],
    [3.13896729e+02, 3.32595917e+02, 3.03317178e-02],
    [3.49918610e+02, 3.80625092e+02, 1.40777826e-01],
    [5.30028015e+02, 4.34657898e+02, 3.38706225e-02],
    [5.70552612e+02, 4.33156982e+02, 6.48662597e-02],
    [4.44476044e+02, 6.13266357e+02, 1.06929235e-01],
    [3.77185120e+02, 4.48166107e+02, 4.42763925e-01],
    [3.45916168e+02, 4.14645752e+02, 8.53749275e-01],
    [3.31532440e+02, 3.42727051e+02, 9.55134451e-01],
    [3.25153564e+02, 2.63553986e+02, 9.44092870e-01],
    [4.88002472e+02, 4.43663361e+02, 8.59685123e-01],
    [5.21022522e+02, 6.08763611e+02, 3.20576370e-01],
    [4.45976929e+02, 5.99758179e+02, 1.87963456e-01],
    [2.26843857e+02, 4.46665192e+02, 8.47813487e-01],
    [1.60803757e+02, 4.67677948e+02, 1.34113967e-01],
    [1.72811050e+02, 4.58672485e+02, 7.52470940e-02]
]])

# Extract only x and y coordinates (ignoring the third value)
keypoints_xy = keypoints_2d[:, :, :2]
new = keypoints_xy[0]

# Extract X and Y values separately
x_values = new[:, 0].flatten().tolist()  # All X values
# y_values = keypoints_xy[1].flatten().tolist()  # All Y values

# Create the desired JSON structure
json_structure = {
    "Human 3d pose data": {
        "Human pose data": {},
        "3d pose axis data": {
            "X-axis coordinate": x_values,
            # "Y-axis coordinate": y_values
        }
    }
}

# Convert to JSON string for visualization
json_output = json.dumps(json_structure, indent=4)

# Display the JSON structure
print(json_output)