# RTE test case running the state estimator.

import logging
from pyvolt import network
from pyvolt import nv_state_estimator
from pyvolt import measurement
import cimpy
import os
import pandas as pd

logging.basicConfig(filename='run_nv_state_estimator.log', level=logging.INFO, filemode='w')

this_file_folder = os.path.dirname(os.path.realpath(__file__))
xml_path = os.path.realpath(os.path.join(this_file_folder, "..", "sample_data", "CIGRE-HV"))
xml_files = [os.path.join(xml_path, "14N_SSH.xml"),
             os.path.join(xml_path, "14N_EQ.xml"),
             os.path.join(xml_path, "14N_SV.xml"),
             os.path.join(xml_path, "14N_TP.xml")]

# Read cim files and create new network.System object
res = cimpy.cim_import(xml_files, "cgmes_v2_4_15")
system = network.System()
base_apparent_power = 10  # MW
system.load_cim_data(res['topology'], base_apparent_power)

# Create measurements data structures
measurements_set = measurement.MeasurementSet()
l_meas = pd.read_csv(os.path.join(xml_path, "Measures_14N_full.csv")).values.tolist()

'''
# voltage/phase measures
for node in system.nodes:
    l_node = [elem[4] for elem in l_meas if elem[6] == node.uuid]
    if l_node:
        node_p = l_node[0]
    else:
        node_p = 0
    measurements_set.create_measurement(node, measurement.ElemType.Node, measurement.MeasType.Vpmu_mag, node_p, 5)
    measurements_set.measurements[-1].meas_value =measurements_set.measurements[-1].meas_value_ideal
for node in system.nodes:
    l_node = [elem[5] for elem in l_meas if elem[6] == node.uuid]
    if l_node:
        node_p = l_node[0]
    else:
        node_p = 0
    measurements_set.create_measurement(node, measurement.ElemType.Node, measurement.MeasType.Vpmu_phase, node_p, 5)
    measurements_set.measurements[-1].meas_value = measurements_set.measurements[-1].meas_value_ideal
'''

#'''
# P/Q measures
for node in system.nodes:
    l_node = [elem[2] for elem in l_meas if elem[6] == node.uuid]
    if l_node:
        node_p = l_node[0]
    else:
        node_p = 0
    measurements_set.create_measurement(node, measurement.ElemType.Node, measurement.MeasType.Sinj_real, node_p, 5)
    measurements_set.measurements[-1].meas_value = measurements_set.measurements[-1].meas_value_ideal
for node in system.nodes:
    l_node = [elem[3] for elem in l_meas if elem[6] == node.uuid]
    if l_node:
        node_p = l_node[0]
    else:
        node_p = 0
    measurements_set.create_measurement(node, measurement.ElemType.Node, measurement.MeasType.Sinj_imag, node_p, 5)
    measurements_set.measurements[-1].meas_value = measurements_set.measurements[-1].meas_value_ideal
#'''

# Perform state estimation
state_estimation_results = nv_state_estimator.DsseCall(system, measurements_set, "advanced")
#state_estimation_results = nv_state_estimator.DsseCall(system, measurements_set)

# Print node voltages
print("state_estimation_results.voltages: ")
for node in state_estimation_results.nodes:
    print('{}={}'.format(node.topology_node.uuid, node.voltage))
