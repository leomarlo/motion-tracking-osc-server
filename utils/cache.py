import numpy as np

Cache = {
      "center":{
        "previous": np.array([0, 0]), 
        "current": np.array([0, 0])
        },
      "size": {
        "previous": 0,
        "current": 0
      },
      "time": {
        "previous": 0,
        "current": 0
      },
      "mobiletime": {
        "previous": 0,
        "current": 0
      },
      "gyroEnergyDeriv": {
        "alpha": 0.1,
        "oldGyroEnergyValue": 0,
        "oldGyroEnergyDerivEWMA": 0,
        "oldScaledGyroEWADerivative": 0,
      },
      "gyroEnergy": {
        "alpha": 0.015,
        "oldEWAEnergy": 0,
        "newEWAEnergy": 0
      }, 
      "max":{
        "sizeVel": 0,
        "centerVel": 0,
        "size": 0,
        "gyroEnergy":12,
        "gyroEnergyEWADeriv": 4 * (10 **(-16)),
        "gyroEnergyDeriv": 3 * (10**(-15))
      },
      "min":{
        "centerVel": 0,
        "sizeVel": 0,
        "size": 0,
        "gyroEnergy":2,
        "gyroEnergyEWADeriv": 0,
        "gyroEnergyDeriv": 10**(-17)
      },
      "MIDI_MAX": 127,
      "SIGMOID_MAX": 5,
      "CAPTURE_INDEX": 0,
      "SHOW_WINDOW": True
    }