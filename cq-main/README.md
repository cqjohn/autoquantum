# Central Quantum Repository

Data directory contains John's data from two experiments, see Readme in that directory for more information


FROM QFLOW-SUITE (https://github.com/jpzwolak/QFlow-suite/blob/master/QFlow-2.0/QFlow):
Crop_Data.py: 
  crop_full_dataset:
    data_key = "noisy_sensor" -> "sensor"
  with h5py.File(h5_filename, 'r') as h5f:...
    v['noise_mag_factor'] -> v['noise_level']
