import shutil

def set_XML_obj(path,num_obj):
    """
    Copies and modifies an XML file to configure objects in a simulation environment.

    This function duplicates a template XML file, appends object definitions to it, 
    and finalizes the XML structure. Each object is defined with specific attributes 
    such as position, geometry, and material properties.

    Args:
        path (str): The directory path where the XML template file (`object_v2pattern.xml`) 
            is located and where the modified XML file (`object_v2.xml`) will be saved.
        num_obj (int): The number of objects to be added to the XML file.

    Raises:
        FileNotFoundError: If the template XML file (`object_v2pattern.xml`) does not exist 
            in the specified path.
        IOError: If there is an issue with reading or writing the XML file.

    Notes:
        - The function assumes that the template XML file (`object_v2pattern.xml`) exists 
            in the specified path.
        - The appended object definitions include attributes such as position, quaternion, 
            joint type, inertial properties, and multiple geometries for visualization and collision.
        - The XML structure is finalized by appending closing tags for `<worldbody>` and `<mujoco>`.

    Example:
        set_XML_obj("/home/user/simulation/", 5)
        This will create a new XML file with 5 object definitions based on the template.
    """

    shutil.copyfile(path+"object_v2pattern.xml",path+"object_v2.xml")
    file = open(path+"object_v2.xml","a+")

    for i in range(num_obj):
        file.write(f"\
        <body name=\"object{i}\" pos=\"-0.35 0.35 0.21\" quat=\"1 0 0 0\">\n \
        <site name=\"site_obj{i}\" pos=\"0.002 -0.013 0.0\" size=\"0.017\" material=\"highlight\" type=\"sphere\"/>\n \
        <joint type=\"free\" limited = \"auto\" armature=\"0.000001\"/> \n \
        <inertial pos=\"0 0 0\" mass=\"0.1\" diaginertia=\"0.0001 0.0001 0.0001\"/>\n \
        <geom name = \"obj{i}\" material=\"Iron_Scratched\" mesh=\"object\" class=\"visual1\"/>\n\
        <geom mesh=\"object_collision_0\" class=\"collision1\"/>\n\
        <geom mesh=\"object_collision_1\" class=\"collision1\" margin=\"0.00075\"/>\n\
        <geom mesh=\"object_collision_2\" class=\"collision1\" margin=\"0.00075\"/>\n\
        <geom mesh=\"object_collision_3\" class=\"collision1\"/>\n\
        <geom mesh=\"object_collision_4\" class=\"collision1\"/>\n\
        </body>\n\n")

    file.write("\n</worldbody>\n</mujoco>")  
    file.close()