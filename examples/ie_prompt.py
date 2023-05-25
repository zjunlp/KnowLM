relation_template =  {
    0:'已知候选的关系列表：{s_schema}，请你根据关系列表，从以下输入中抽取出可能存在的头实体与尾实体，并给出对应的关系三元组。请按照{s_format}的格式回答。',
    1:'我将给你个输入，请根据关系列表：{s_schema}，从输入中抽取出可能包含的关系三元组，并以{s_format}的形式回答。',
    2:'我希望你根据关系列表从给定的输入中抽取可能的关系三元组，并以{s_format}的格式回答，关系列表={s_schema}。',
    3:'给定的关系列表是：{s_schema}\n根据关系列表抽取关系三元组，在这个句子中可能包含哪些关系三元组？请以{s_format}的格式回答。',
}

relation_int_out_format = {
    0:['"(头实体,关系,尾实体)"', relation_convert_target0],
    2:['"关系：头实体,尾实体\n"', relation_convert_target2],
    3:["JSON字符串[{'head':'', 'relation':'', 'tail':''}, ]", relation_convert_target3],
}
