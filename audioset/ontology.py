import json
import os


class ASOntologyNode(object):
    def __init__(self, ontology, _id, name, description, citation_uri, positive_examples, child_ids, restrictions):
        self.ontology = ontology
        self.id = _id
        self.name = name
        self.description = description
        self.citation_uri = citation_uri
        self.positive_examples = positive_examples
        self.child_ids = child_ids
        self.restrictions = restrictions

        # Set restrictions
        self.abstract = False
        self.blacklist = False
        for restr in restrictions:
            if restr == 'abstract':
                self.abstract = True
            elif restr == 'blacklist':
                self.blacklist = True

        self.parent_id = None

    @property
    def children(self):
        """
        Get a list of this node's children
        """
        return self.ontology.get_node_children(self)

    @property
    def parent(self):
        if self.parent_id is None:
            return None
        else:
            return self.ontology.get_node(self.parent_id)

    def is_child(self, q_child):
        q_child = self.ontology.ensure_node(q_child)
        # DFS to find child
        for child in self.children:
            if child == q_child:
                return True
            elif child.is_child(q_child):
                return True
        return False

    def is_parent(self, q_parent):
        q_parent = self.ontology.ensure_node(q_parent)
        return q_parent.is_child(self)



class ASOntology(object):
    def __init__(self, ontology_path):
        self._nodes = {}
        self._node_name_to_id = {}

        # Make sure the path to the ontology file exists
        if not os.path.exists(ontology_path):
            error_msg =  'Cannot find ontology at "{}"'
            raise ValueError(error_msg.format(ontology_path))

        # Load the ontology object
        with open(ontology_path, 'r') as f:
            ontology_list = json.load(f)

        # Create ontology nodes
        for ontology_item in ontology_list:
            _id = ontology_item['id']
            node = ASOntologyNode(
                self,
                _id,
                ontology_item['name'],
                ontology_item['description'],
                ontology_item['citation_uri'],
                ontology_item['positive_examples'],
                ontology_item['child_ids'],
                ontology_item['restrictions']
            )
            self._nodes[_id] = node

        self._init_tree()

    def _init_tree(self):

        # Do a pass through the node to assign parents
        for node in self._nodes.values():
            for child in node.children:
                child.parent_id = node.id

            # (Also use this loop to reate a mapping from node name to ID)
            self._node_name_to_id[node.name] = node.id

        # Do another pass to find the top level nodes, which have not
        # been assigned a parent
        self.top_level_node_ids = []
        for node in self._nodes.values():
            if node.parent is None:
                self.top_level_node_ids.append(node.id)

    @property
    def top_level_nodes(self):
        return [self.get_node(node_id) for node_id in self.top_level_node_ids]

    def ensure_node(self, node):
        if not isinstance(node, ASOntologyNode):
            node = self._nodes[node]

        return node

    def get_node_children(self, node):
        """
        Get a list of children nodes of a given node
        """
        node = self.ensure_node(node)

        return [self._nodes[child_id] for child_id in node.child_ids]

    def get_node(self, node_id):
        """
        Get a node with the given id
        """
        if node_id not in self._nodes:
            raise ValueError('No node with ID {}'.format(node_id))
        return self._nodes[node_id]

    def get_node_by_name(self, node_name):
        """
        Get a node with the given name
        """
        if node_name not in self._node_name_to_id:
            raise ValueError('No node with name {}'.format(node_name))
        return self.get_node(self._node_name_to_id[node_name])


