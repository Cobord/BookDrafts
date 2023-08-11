"""
test Plabic graphs
"""
#pylint:disable=import-error,too-many-locals
from plabic import PlabicGraph,BiColor

def test_plabic() -> None:
    """
    example from figure 7.1 of https://arxiv.org/pdf/2106.02160.pdf
    black there -> red here
    white there -> green here
    the names of internal vertices are numbered top to bottom
    left to right and noting the one attached to external 6
    is slightly below the 3rd and 4th (though that is hard to see in the picture)
    """
    external_orientation = ["ext1", "ext2", "ext3", "ext4", "ext5", "ext6"]
    _internal_vertices = ["int1", "int2", "int3",
                         "int4", "int5", "int6", "int7", "int8"]
    my_data = {}
    my_data["ext1"] = (BiColor.RED, ["int1"])
    my_data["ext2"] = (BiColor.RED, ["int2"])
    my_data["ext3"] = (BiColor.RED, ["int7"])
    my_data["ext4"] = (BiColor.RED, ["int8"])
    my_data["ext5"] = (BiColor.RED, ["int6"])
    my_data["ext6"] = (BiColor.RED, ["int5"])

    my_data["int1"] = (BiColor.GREEN, ["ext1", "int2", "int3"])
    my_data["int2"] = (BiColor.RED, ["int1", "ext2", "int4"])
    my_data["int3"] = (BiColor.RED, ["int1", "int4", "int6"])
    my_data["int4"] = (BiColor.GREEN, ["int2", "int7", "int3"])
    my_data["int5"] = (BiColor.GREEN, ["ext6"])
    my_data["int6"] = (BiColor.GREEN, ["int3", "int8", "ext5"])
    my_data["int7"] = (BiColor.GREEN, ["int4", "ext3", "int8"])
    my_data["int8"] = (BiColor.RED, ["int6", "int7", "ext4"])
    example_plabic = PlabicGraph(my_data, external_orientation, {})
    expected_one_steps = ["int1", "int2", "int7", "int8", "int6", "int5"]
    expected_two_steps = ["int2", "int1", "int8", "int7", "int3", "ext6"]
    for cur_bdry_temp, exp_one_step, exp_two_step in\
            zip(external_orientation, expected_one_steps, expected_two_steps):
        one_step = example_plabic.follow_rules_of_road(cur_bdry_temp, None, None)
        assert one_step is not None
        cur_one_step_idx, cur_one_step = one_step
        assert cur_one_step == exp_one_step
        two_step = example_plabic.follow_rules_of_road(
            cur_one_step, cur_bdry_temp, cur_one_step_idx)
        assert two_step is not None
        assert two_step[1] == exp_two_step
    expected_perm = {"ext1":"ext3", "ext2":"ext4",
                         "ext3":"ext5","ext4":"ext1",
                         "ext5":"ext2","ext6":"ext6"}
    expected_decorated = {"ext6":BiColor.GREEN}
    for cur_bdry_temp in external_orientation:
        turn_color, cur_tgt = example_plabic.bdry_to_bdry(cur_bdry_temp)
        assert cur_tgt == expected_perm[cur_bdry_temp]
        assert turn_color == expected_decorated.get(cur_bdry_temp,None)
    changed, explanation = example_plabic.square_move(("int1", "int2", "int4", "int3"))
    assert changed and explanation == "Success"
    for cur_bdry_temp in external_orientation:
        turn_color, cur_tgt = example_plabic.bdry_to_bdry(cur_bdry_temp)
        assert cur_tgt == expected_perm[cur_bdry_temp]
        assert turn_color == expected_decorated.get(cur_bdry_temp,None)
    changed, explanation = example_plabic.square_move(("int1", "int2", "int4", "int3"))
    assert changed and explanation == "Success"
    for cur_bdry_temp in external_orientation:
        turn_color, cur_tgt = example_plabic.bdry_to_bdry(cur_bdry_temp)
        assert cur_tgt == expected_perm[cur_bdry_temp]
        assert turn_color == expected_decorated.get(cur_bdry_temp,None)
    changed, explanation = example_plabic.flip_move("int4", "int7")
    assert changed and explanation == "Success"
    for cur_bdry_temp in external_orientation:
        turn_color, cur_tgt = example_plabic.bdry_to_bdry(cur_bdry_temp)
        assert cur_tgt == expected_perm[cur_bdry_temp]
        assert turn_color == expected_decorated.get(cur_bdry_temp,None)
