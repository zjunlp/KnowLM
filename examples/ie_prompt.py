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


en_relation_template = {
    0: 'Identify the head entities (subjects) and tail entities (objects) in the following text and provide the corresponding relation triples from relation list {s_schema}. Please provide your answer as a list of relation triples in the form of {s_format}.',
    1: 'Identify the subjects and objects in the text that are related, and provide the corresponding relation triples from relation {s_schema} in the format of {s_format}.',
    2: 'From the given text, extract the possible head entities (subjects) and tail entities (objects) and give the corresponding relation triples. The relations are {s_schema}. Please format your answer as a list of relation triples in the form of {s_format}.',
    3: 'Your task is to identify the head entities (subjects) and tail entities (objects) in the following text and extract the corresponding relation triples, the possible relation list is {s_schema}. Your answer should include relation triples, with each triple formatted as {s_format}.',
    4: 'Given the text, extract the possible head entities (subjects) and tail entities (objects) and provide the corresponding relation triples, the possible relation list is {s_schema}. Format your answer as a list of relation triples in the form of {s_format}.',
    5: 'Your goal is to identify the head entities (subjects) and tail entities (objects) in the text and give the corresponding relation triples. The given relation list is {s_schema}. Please answer with a list of relation triples in the form of {s_format}.',
    6: 'Please extract the possible head entities (subjects) and tail entities (objects) from the text and provide the corresponding relation triples from candidate relation list {s_schema}. Your answer should be in the form of a list of relation triples: {s_format}.',
    7: 'Your task is to extract the possible head entities (subjects) and tail entities (objects) in the given text and give the corresponding relation triples. The relations are {s_schema}. Please answer using the format of a list of relation triples: {s_format}.',
    8: 'Given the {s_schema}, identify the head entities (subjects) and tail entities (objects) and provide the corresponding relation triples. Your answer should consist of relation triples, with each triple formatted as {s_format}',
    9: 'Please find the possible head entities (subjects) and tail entities (objects) in the text based on the relation list {s_schema} and give the corresponding relation triples. Please format your answer as a list of relation triples in the form of {s_format}.',
    10: 'Given relation list {s_schema}, extract the possible subjects and objects from the text and give the corresponding relation triples in the format of {s_format}.',
    11: 'Extract the entities involved in the relationship described in the text and provide the corresponding triples in the format of {s_format}, the possible relation list is {s_schema}.',
    12: 'Given relation list {s_schema}, provide relation triples for the entities and their relationship in the text, using the format of {s_format}.',
    13: 'Extract the entities and their corresponding relationships from the given relationships are {s_schema} and provide the relation triples in the format of {s_format}.',
}

en_relation_int_out_format = {
    0: "{'head':'', 'relation':'', 'tail':''}",
    1: "(Subject, Relation, Object)",
    2: "[Subject, Relation, Object]",
    3: "{head, relation, tail}",
    4: "<head, relation, tail>",
}
