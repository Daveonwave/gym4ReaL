import shutil

def set_XML_obj(path,num_obj):

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