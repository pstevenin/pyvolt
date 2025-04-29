import networkx as nx
import numpy as np
from enum import Enum

#function to find node root
def find_root(node_name, connections):
    while connections[node_name] != node_name:
        node_name = connections[node_name]
    return node_name

class BusType(Enum):
    SLACK = 1
    slack = 1
    PV = 2
    pv = 2
    PQ = 3
    pq = 3


class Node():
    def __init__(self, uuid='', name='', base_voltage=1.0, base_apparent_power=1.0, v_mag=0.0,
                 v_phase=0.0, p=0.0, q=0.0, index=0, ideal_connected_with='a'):
        self.uuid = uuid
        self.name = name
        self.index = index
        self.baseVoltage = base_voltage
        self.base_apparent_power = base_apparent_power
        self.base_current = self.base_apparent_power / self.baseVoltage / np.sqrt(3)
        self.type = BusType["PQ"]
        self.voltage = complex(v_mag * np.cos(np.radians(v_phase)), v_mag * np.sin(np.radians(v_phase)))
        self.power = complex(p, q)
        self.power_pu = complex(p, q) / self.base_apparent_power
        self.voltage_pu = self.voltage / self.baseVoltage
        self.ideal_connected_with = ideal_connected_with

    def __str__(self):
        string = 'class=Node\n'
        attributes = self.__dict__
        for key in attributes.keys():
            string = string + key + '={}\n'.format(attributes[key])
        return string
        

class Branch():
    def __init__(self, uuid='', r=0.0, x=0.0, start_node=None, end_node=None,
                 base_voltage=1.0, base_apparent_power=1.0):
        self.uuid = uuid
        self.baseVoltage = base_voltage
        self.base_apparent_power = base_apparent_power
        self.base_current = self.base_apparent_power / self.baseVoltage / np.sqrt(3)
        self.base_impedance = base_voltage ** 2 / self.base_apparent_power
        self.start_node = start_node
        self.end_node = end_node
        self.r = r
        self.x = x
        self.z = self.r + 1j * self.x
        self.y = 1 / self.z if (self.z != 0) else float("inf")
        self.r_pu = r / self.base_impedance
        self.x_pu = x / self.base_impedance
        self.z_pu = self.r_pu + 1j * self.x_pu
        self.y_pu = 1 / self.z_pu if (self.z_pu != 0) else float("inf")

    def __str__(self):
        string = 'class=Branch\n'
        attributes = self.__dict__
        for key in attributes.keys():
            string = string + key + '={}\n'.format(attributes[key])
        return string


class Breaker():
    def __init__(self, from_node, to_node, is_open=True):
        """
        :param from_node:
        :param to_node:
        :param is_open: True if the breaker is considered open and False if the broker is closed 
        """
        self.from_node = from_node
        self.to_node = to_node
        self.is_open = is_open

    def __str__(self):
        string = 'class=Breaker\n'
        attributes = self.__dict__
        for key in attributes.keys():
            string = string + key + '={}\n'.format(attributes[key])
        return string


class System():
    def __init__(self):
        self.nodes = []
        self.branches = []
        self.breakers = []
        self.Ymatrix = np.zeros([], dtype=np.complex128)
        self.Bmatrix = np.zeros([], dtype=np.complex128)

    def get_node_by_uuid(self, node_uuid):
        for node in self.nodes:
            if node.uuid == node_uuid:
                return node
        
        return False

    def get_node_by_index(self, index):
        """
        Return the node with node.index == index
        """
        for node in self.nodes:
            if (node.index == index) and (node.ideal_connected_with == ''):
                return node
        
        return None
           
    def get_nodes_num(self):
        """
        Return the number of nodes in the list system.nodes
        Warning: if any node is ideally connected to another node, 
        the counter is increased only one time
        """
        nodes_num = 0
        for node in self.nodes:
            if node.ideal_connected_with == '':
                nodes_num += 1

        return nodes_num

    def reassign_connected(self):
        #group nodes per index
        grouped_objects = {}
        for node in self.nodes:
            idx = node.index
            if idx not in grouped_objects:
                grouped_objects[idx] = []
            grouped_objects[idx].append(node)

        #attribute ideal_connected_with to each group
        for idx, group in grouped_objects.items():
            found = False
            for obj in group:
                if obj.type.name in ["PQ", "SLACK"]:
                    obj.ideal_connected_with = ""
                    found = True
                    break
            if not found:
                group[0].ideal_connected_with = ""
             
    def load_cim_data(self, res, base_apparent_power):
        """
        fill the vectors node, branch and breakers
        """
        self.nodes = []
        self.branches = []
        self.breakers = []

        index = 0
        list_TPNode = [elem for elem in res.values() if elem.__class__.__name__ == "TopologicalNode"]
        list_SvVoltage = [elem for elem in res.values() if elem.__class__.__name__ == "SvVoltage"]
        list_SvPowerFlow = [elem for elem in res.values() if elem.__class__.__name__ == "SvPowerFlow"]
        list_EnergySources = [elem for elem in res.values() if elem.__class__.__name__ == "EnergySource"]
        list_EnergyConsumer = [elem for elem in res.values() if elem.__class__.__name__ == "EnergyConsumer"]
        list_ACLineSegment = [elem for elem in res.values() if elem.__class__.__name__ == "ACLineSegment"]
        list_PowerTransformer = [elem for elem in res.values() if elem.__class__.__name__ == "PowerTransformer"]
        list_Terminals = [elem for elem in res.values() if elem.__class__.__name__ == "Terminal"]
        list_Terminals_ES = [elem for elem in list_Terminals if
                             elem.ConductingEquipment.__class__.__name__ == "EnergySource"]
        list_Terminals_EC = [elem for elem in list_Terminals if
                             elem.ConductingEquipment.__class__.__name__ == "EnergyConsumer"]
        list_PowerTransformerEnds = [elem for elem in res.values() if elem.__class__.__name__ == "PowerTransformerEnd"]
        list_Breakers = [elem for elem in res.values() if elem.__class__.__name__ == "Breaker"]
        list_DCLineSegment = [elem for elem in res.values() if elem.__class__.__name__ == "DCLineSegment"]
        list_DCTerminals = [elem for elem in res.values() if elem.__class__.__name__ == "DCTerminal"]
        list_VsConverter = [elem for elem in res.values() if elem.__class__.__name__ == "VsConverter"]

        #dictionary to get voltage from element name
        value_map = {1: 5, 2: 20, 3: 63, 4: 90, 5: 150, 6: 225, 7: 400, 9: 320}

        #create nodes
        for TPNode in list_TPNode:
            uuid_TPNode = TPNode.mRID
            name = TPNode.name
            vmag = 0.0
            vphase = 0.0
            pInj = 0.0
            qInj = 0.0
                
            for obj_SvVoltage in list_SvVoltage:
                if obj_SvVoltage.TopologicalNode.mRID == uuid_TPNode:
                    vmag = obj_SvVoltage.v
                    vphase = obj_SvVoltage.angle
                    break
            for obj_SvPowerFlow in list_SvPowerFlow:
                if (obj_SvPowerFlow.Terminal.TopologicalNode.mRID == uuid_TPNode and
                        obj_SvPowerFlow.Terminal.ConductingEquipment.__class__.__name__ != "VsConverter"):
                    pInj -= obj_SvPowerFlow.p
                    qInj -= obj_SvPowerFlow.q
            for obj_Terminal in list_Terminals_ES:
                if obj_Terminal.TopologicalNode.mRID == uuid_TPNode:
                    for obj_EnergySource in list_EnergySources:
                        if obj_EnergySource.mRID == obj_Terminal.ConductingEquipment.mRID:
                            pInj += obj_EnergySource.activePower
                            qInj += obj_EnergySource.reactivePower
            for obj_Terminal in list_Terminals_EC:
                if obj_Terminal.TopologicalNode.mRID == uuid_TPNode:
                    for obj_EnergyConsumer in list_EnergyConsumer:
                        if obj_EnergyConsumer.mRID == obj_Terminal.ConductingEquipment.mRID:
                            pInj -= obj_EnergyConsumer.p
                            qInj -= obj_EnergyConsumer.q

            #get nominal voltage from substation name
            if TPNode.BaseVoltage is None:
                base_voltage = value_map.get(int(TPNode.name[-1]), None)
            else:
                base_voltage = TPNode.BaseVoltage.nominalVoltage
            self.nodes.append(Node(name=name, uuid=uuid_TPNode, base_voltage=base_voltage, v_mag=vmag,
                                   base_apparent_power=base_apparent_power, v_phase=vphase,
                                   p=pInj, q=qInj, index=index))
            index = index + 1
        
        self._setNodeType(list_Terminals)   

        #create branches type ACLineSegment
        for ACLineSegment in list_ACLineSegment:
            uuid_ACLineSegment = ACLineSegment.mRID
            nodes = self._get_nodes(list_Terminals, uuid_ACLineSegment)
            start_node = nodes[0]
            end_node = nodes[1]

            #exclude line connected to one node with no voltage
            if nodes[0] == False or nodes[1] == False:
                continue

            #get nominal voltage from line name
            if ACLineSegment.BaseVoltage is None:
                if ACLineSegment.name[6] in ["B", "T", "G", "P"]:
                    base_voltage = value_map.get(int(ACLineSegment.name[5]), None)
                else:
                    base_voltage = value_map.get(int(ACLineSegment.name[6]), None)
            else:
                base_voltage = ACLineSegment.BaseVoltage.nominalVoltage
            self.branches.append(Branch(uuid=uuid_ACLineSegment, r=ACLineSegment.r, x=ACLineSegment.x, 
                                        start_node=start_node, end_node=end_node, 
                                        base_voltage=base_voltage, base_apparent_power=base_apparent_power))

            #create branches type DCline
            for DCLineSegment in list_DCLineSegment:
                uuid_DCLineSegment = DCLineSegment.mRID
                start_DCEq = [elem.DCNode.DCEquipmentContainer.mRID for elem in list_DCTerminals if
                              elem.DCConductingEquipment.mRID != DCLineSegment.mRID and elem.sequenceNumber == 1][0]
                end_DCEeq = [elem.DCNode.DCEquipmentContainer.mRID for elem in list_DCTerminals if
                             elem.DCConductingEquipment.mRID != DCLineSegment.mRID and elem.sequenceNumber == 2][0]
                start_VsConv = [elem.mRID for elem in list_VsConverter if elem.EquipmentContainer.mRID == start_DCEq][0]
                end_VsConv = [elem.mRID for elem in list_VsConverter if elem.EquipmentContainer.mRID == end_DCEeq][0]
                start_uuid = [elem.TopologicalNode.mRID for elem in list_Terminals if
                              elem.ConductingEquipment.mRID == start_VsConv][0]
                end_uuid = [elem.TopologicalNode.mRID for elem in list_Terminals if
                            elem.ConductingEquipment.mRID == end_VsConv][0]
                start_node = self.get_node_by_uuid(start_uuid)
                end_node = self.get_node_by_uuid(end_uuid)

                #exclude branch connected to at least one node with no voltage
                if start_node == False or end_node == False:
                    continue

                #get nominal voltage from line name
                if DCLineSegment.name[6] in ["B", "T", "G", "P"]:
                    base_voltage = value_map.get(int(DCLineSegment.name[5]), None)
                else:
                    base_voltage = value_map.get(int(DCLineSegment.name[6]), None)

                #TO DO: insert converter station losses
                self.branches.append(
                    Branch(uuid=uuid_DCLineSegment, r=DCLineSegment.resistance, x=DCLineSegment.inductance,
                           start_node=start_node, end_node=end_node,
                           base_voltage=base_voltage, base_apparent_power=base_apparent_power))

        #create branches type powerTransformer
        for power_transformer in list_PowerTransformer:
            uuid_power_transformer = power_transformer.mRID
            nodes = self._get_nodes(list_Terminals, uuid_power_transformer)
            start_node = nodes[0]
            end_node = nodes[1]

            #exclude transformer connected to one node with no voltage
            if nodes[0] == False or nodes[1] == False:
                continue
            #base voltage = high voltage side (=primaryConnection)
            primary_connection = self._get_primary_connection(list_PowerTransformerEnds, uuid_power_transformer)
            #get nominal voltage from transformer name
            if primary_connection.BaseVoltage is None:
                if power_transformer.name[6] == "A" or power_transformer.name[6] == "T":
                    base_voltage = value_map.get(int(power_transformer.name[10]), None)
                else:
                    base_voltage = value_map.get(int(power_transformer.name[6]), None)
            else:
                base_voltage = primary_connection.BaseVoltage.nominalVoltage
            self.branches.append(Branch(uuid=uuid_power_transformer, r=primary_connection.r, x=primary_connection.x,
                                        start_node=start_node, end_node=end_node, base_voltage=base_voltage,
                                        base_apparent_power=base_apparent_power))

        #create breakers
        for obj_Breaker in list_Breakers:
            is_open = obj_Breaker.open
            nodes = self._get_nodes(list_Terminals, obj_Breaker.mRID)
            #exclude breaker connected to one node with no voltage
            if nodes[0] == False or nodes[1] == False:
                continue
            self.breakers.append(Breaker(from_node=nodes[0], to_node=nodes[1], is_open=is_open))

        #create graph from nodes, branches and breakers
        graph_nx = nx.Graph()
        for node in self.nodes:
            graph_nx.add_node(node.index, name=node.uuid)
        for branch in self.branches:
            graph_nx.add_edge(branch.start_node.index, branch.end_node.index)
        for breaker in self.breakers:
            if not breaker.is_open:  #add non broken links only
                graph_nx.add_edge(breaker.from_node.index, breaker.to_node.index)

        #find related component
        components = list(nx.connected_components(graph_nx))

        #find largest related component
        largest_component = max(components, key=len)

        #find disconnected nodes
        nodes_to_remove = set(graph_nx.nodes()) - largest_component
        graph_nx.remove_nodes_from(nodes_to_remove)

        #delete nodes, branches and breakers disconnected from network
        self.nodes = [b for b in self.nodes if b.index in graph_nx]
        self.branches = [b for b in self.branches if
                         b.start_node.index in graph_nx and b.end_node.index in graph_nx]
        self.breakers = [b for b in self.breakers if b.from_node.index in graph_nx and b.to_node.index in graph_nx]

        #dictionary to stock connections
        connections = {node.uuid: node.uuid for node in self.nodes}

        #new indices: connected nodes have the same index, depending on breakers position
        for breaker in self.breakers:
            if not breaker.is_open:
                root_from = find_root(breaker.from_node.uuid, connections)
                root_to = find_root(breaker.to_node.uuid, connections)
                if root_from != root_to:
                    connections[root_to] = root_from

        #update indices in self.nodes
        for node in self.nodes:
            root = find_root(node.uuid, connections)
            node.index = [elem for elem in self.nodes if elem.uuid == root][0].index

        #reassign indices to be consecutive
        new_index = {}
        index_count = 0
        for item in self.nodes:
            if item.index not in new_index:
                new_index[item.index] = index_count
                index_count += 1
            item.index = new_index[item.index]

        #reassign ideal_connected_with parameter to get correct node type
        self.reassign_connected()

        #calculate admittance matrix
        self.Ymatrix_calc()

    def _get_nodes(self, list_Terminals, elem_uuid):
        """
        get the the start and end node of the element with uuid=elem_uuid
        This function can used only with element which are connected 
        to 2 topologicalNodes, for example: ACLineSegment, PowerTransformer and Breaker 
        :param list_Terminals: list of all elements of type Terminal
        :param elem_uuid: uuid of the element for which the start and end node ID are searched
        :return list: [startNodeID, endNodeID]
        """
        start_node_uuid = None
        end_node_uuid = None
        
        for terminal in list_Terminals:
            if terminal.ConductingEquipment.mRID != elem_uuid:
                continue
            sequence_number = terminal.sequenceNumber
            if sequence_number == 1:
                start_node_uuid = terminal.TopologicalNode.mRID
            elif sequence_number == 2:
                end_node_uuid = terminal.TopologicalNode.mRID
        
        start_node = None
        end_node = None
        if start_node_uuid is None:
            print('WARNING: It could not find a start node for the element with uuid={}'.format(elem_uuid))
        else:
            start_node = self.get_node_by_uuid(start_node_uuid)
        if end_node_uuid is None:
            print('WARNING: It could not find a end node for the element with uuid={}'.format(elem_uuid))
        else:
            end_node = self.get_node_by_uuid(end_node_uuid)

        return [start_node, end_node]

    def _get_primary_connection(self, list_PowerTransformerEnds, elem_uuid):
        """
        get primaryConnection of the powertransformer with uuid = elem_uuid
        :param list_PowerTransformerEnds: list of all elements of type PowerTransformerEnd
        :param elem_uuid: uuid of the power transformer for which the primary connection is searched
        :return: primary_connection
        """
        power_transformer_ends = []

        #search for two elements of class powertransformerend that point to the powertransformer with ID == elem_uuid
        for power_transformer_end in list_PowerTransformerEnds:
            if isinstance(power_transformer_end.PowerTransformer, list):
                if len(power_transformer_end.PowerTransformer) != 1:
                    print('WARNING: len(power_transformer_end.PowerTransformer)!=1 for the element with uuid={}. \
                        The first element will be used'.format(power_transformer_end.mRID))
                power_transformer = power_transformer_end.PowerTransformer[0]
            else:
                power_transformer = power_transformer_end.PowerTransformer
        
            if power_transformer.mRID == elem_uuid:
                power_transformer_ends.append(power_transformer_end)

        if power_transformer_ends[0].endNumber == 1:
            primary_connection = power_transformer_ends[0]
        else:
            primary_connection = power_transformer_ends[1]

        return primary_connection

    def _setNodeType(self, list_Terminals):
        """
        set the parameter "type" of all elements of the list self.nodes
        :param list_Terminals: list of all elements of type Terminal
        :return None
        """
        #get a list of Terminals for which the ConductingEquipment is a element of class ExternalNetworkInjection
        list_Terminals_ENI = [elem for elem in list_Terminals if
                              elem.ConductingEquipment.__class__.__name__ == "ExternalNetworkInjection"]
        for terminal in list_Terminals_ENI:
            node_uuid = terminal.TopologicalNode.mRID
            for node in self.nodes:
                if node.uuid == node_uuid:
                    node.type = BusType["SLACK"]

        #get a list of Terminals for which the ConductingEquipment is a element of class SynchronousMachine
        list_Terminals_SM = [elem for elem in list_Terminals
                             if elem.ConductingEquipment.__class__.__name__ == "SynchronousMachine"]
        for terminal in list_Terminals_SM:
            node_uuid = terminal.TopologicalNode.mRID
            for node in self.nodes:
                if node.uuid == node_uuid:
                    node.type = BusType["PV"]

    def Ymatrix_calc(self):
        nodes_num = self.get_nodes_num()
        self.Ymatrix = np.zeros((nodes_num, nodes_num), dtype=np.complex128)
        self.Bmatrix = np.zeros((nodes_num, nodes_num), dtype=np.complex128)
        for branch in self.branches:
            fr = branch.start_node.index
            to = branch.end_node.index
            self.Ymatrix[fr][to] -= branch.y_pu
            self.Ymatrix[to][fr] -= branch.y_pu
            self.Ymatrix[fr][fr] += branch.y_pu
            self.Ymatrix[to][to] += branch.y_pu

    #testing functions
    def print_nodes_names(self):
        for node in self.nodes:
            print('{} {}'.format(node.name, node.index))

    def print_node_types(self):
        for node in self.nodes:
            print('{} {}'.format(node.name, node.type))
    
    def print_power(self):
        for node in self.nodes:
            print('{} {}'.format(node.name, node.power))
