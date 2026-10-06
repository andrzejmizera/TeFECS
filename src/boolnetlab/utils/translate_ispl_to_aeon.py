import argparse
import boolean.boolean as bool

def main(ispl_input_filename, bn_output_filename, network_name_prefix, reorder_variables):

    node_names = []
    rules = []

    with open(ispl_input_filename, "r") as ispl_file:
        l = ispl_file.readline().strip()
        while l != 'Vars:':
            l = ispl_file.readline().strip()

        l = ispl_file.readline().strip()

        while l != 'end Vars':
            if not reorder_variables:    
                node_names.append(l.split(':')[0].strip())
            l = ispl_file.readline().strip()

        while l != 'Evolution:':
            l = ispl_file.readline().strip()

        node_index = 0
        l = ispl_file.readline().strip()
        while l != 'end Evolution':
            
            if l[-5:] == 'true;':

                (node_name_part, rule_part) = l.split(' if ')

                if not reorder_variables:
                    assert node_name_part.split('=')[0].strip() == node_names[node_index], f"{l} node name: {node_names[node_index]}"
                else:
                    node_names.append(node_name_part.split('=')[0].strip())
            
                rule = rule_part.strip().split('=')
                rules.append(rule[0])
            
                node_index += 1

            l = ispl_file.readline().strip()

        assert len(node_names) == len(rules), f"The number of nodes does not match the number of Boolean functions."

        print(f"Number of nodes: {len(node_names)}")

    __bool_algebra = bool.BooleanAlgebra()

    with open(bn_output_filename, "w") as bn_file:
        for node, rule in zip(node_names, rules):
            rule = rule.replace("~", "!") # aeon does not support ~
            rule_definition = "$" + node + ": " + rule  
            set_of_rules = __bool_algebra.parse(rule, simplify=True).literalize().literals  # set of all operands.
            
            for node_with_symbol in set_of_rules:
                node_to_str = str(node_with_symbol)
                subdefinition = ""

                if node_to_str.startswith("~"):
                    node_to_str = node_to_str.replace("~", "", count=1) 
                    subdefinition = subdefinition + node_to_str + " -| " + node  
                else:
                    subdefinition = subdefinition + node_to_str + " -> " + node

                bn_file.write(subdefinition + "\n")
            
            bn_file.write(rule_definition + "\n")
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                    prog='ispl2bn',
                    description='Based on a CABEAN .ispl file, generates two lists of Boolean network nodes and Boolean functions\
                        which are saved in a specified file.',
                    epilog='')
    
    parser.add_argument(dest='input_file', help="The name of the file with .ispl Boolean network specification")
    parser.add_argument(dest='output_file', help="The name of the output file")
    parser.add_argument('-mn', '--model_name', help="The prefix to be used for the names of the variable used to store the\
                         Boolean network specification ", dest="network_name_prefix")
    parser.add_argument('-r', '--reorder', action='store_true', help="Reorder the nodes in accrodance with the order of the\
                         Boolean functions in the .ispl file.")

    args = parser.parse_args()

    ispl_input_filename = args.input_file
    bn_output_filename = args.output_file

    if args.network_name_prefix is not None:
        network_name_prefix = args.network_name_prefix + '_'
    else:
        network_name_prefix = ''

    reorder_variables = False
    if args.reorder is not None:
        reorder_variables = True

    main(ispl_input_filename, bn_output_filename, network_name_prefix, reorder_variables)

