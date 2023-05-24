1. Overview
2. 目录
3. 预训练语料（使用方法，模型合并）
4. 预训练效果
5. 指令微调语料
6. 指令微调效果（抽取、对话）
7. 脚本
8. 未来

## 效果

### 1. 预训练

> [删重] [完成] [未完成] 分别表示对输出重复的部分进行手动截断、模型输出了结束符号、模型因为长度的限制没能输出完成。加粗体为输入模型的prompt，后面为模型的输出。

- [完成]

  **use python to write the bubble sort algorithm.**

  def bubble_sort(arr):
      """Bubble sort algorithm.
      """ for i in range(len(arr) - 1, 0, -1):
          for j in range(len(arr) - 1, 0, -1):
              if arr[j] > arr[j + 1]:
                  arr[j], arr[j + 1] = arr[j + 1], arr[j]
      return arr

- [完成]

  **你非常了解一些健康生活的习惯，请列举几个健康生活的建议：** 1.每天坚持锻炼30分钟以上。  2.不吸烟，不酗酒。  3.少吃高脂肪食物。  4.多吃蔬菜和水果。  5.保证充足的睡眠。  6.保持良好的心情。  7.定期体检。  8.养成良好的卫生习惯。

- [完成]

  **自然语言处理是**计算机科学的一个分支，它研究如何从自然语言中抽取出有意义的信息。 自然语言处理包括词法分析、句法分析、语义分析、语音识别和机器翻译等。 自然语言处理的目标是使计算机能够像人类一样理解自然语言。

- [完成]

  **你对中国的大学了解的非常多。请评价一下浙江大学是一所怎样的学校。** 答:浙江大学是一所历史悠久、学科齐全、学术实力雄厚、师资力量强大、办学特色鲜明,在国内外具有重要影响的综合性研究型大学。 浙江大学是首批进入国家"211工程"和"985工程"建设的高校之一,也是国家"111计划"和"珠峰计划"重点建设的名牌大学。 浙江大学的前身是创建于1897年的求是书院,是中国人最早自己创办的新式高等学府之一。 1928年,求是书院改制为国立浙江大学。 1952年,浙江大学成为新中国创办的第一批全国重点大学。 1998年,浙江大学与杭州大学、浙江农业大学、浙江医科大学、浙江师范大学、浙江师范大学医学院合并组建新的浙江大学。 浙江大学现已发展成为一所包括文、法、理、工、农、医、管、艺等学科门类齐全的综合性研究型大学,是国家"985工程"和"211工程"重点建设的大学之一,也是国家"111计划"和"珠峰计划"重点建设的名牌大学。

- [完成]

  **你很擅长将中文翻译成英文。将下面的句子翻译成英文：我们今天准备去西安看兵马俑。答案：**We are going to see the Terracotta Warriors in Xi'an today.

- [完成]

  **You are good at translating English into Chinese. Translate the following sentence into Chinese: Nothing is difficult to a willing heart. Answer:** 什么都不难,只要心甘情愿。

- [完成]

  **Here is the recommendation letter that I wrote for an application to a dragon feeder position at the Magic Unicorn Corporation:
  Dear recruiter**,
  I am writing this letter of recommendation on behalf of my student, who is applying for a position as a dragon feeder at the Magic Unicorn Corporation.  I have had the pleasure of teaching this student for the past three years, and I can say without hesitation that he is one of the best students I have ever had.  He is hard-working, diligent, and intelligent.  He is also a very good dragon feeder.  In fact, he is the best dragon feeder I have ever had in my class.  He has fed many dragons over the years, and they have all been very pleased with his work.  I would not hesitate to recommend him for a position as a dragon feeder at the Magic Unicorn Corporation.
   Thank you for your consideration.
   Sincerely,
  Professor Snape
  P.S.  Please don't tell anyone that I wrote this letter.  I don't want to get into any trouble with the Ministry of Magic.

   \# Chapter 1

  \# THE MAGIC UNICORN CORPORATION

  The Magic Unicorn Corporation (MUC) is a large multinational corporation headquartered in New York City.  The company was founded in 1980 by a group of unicorns who wanted to start their own business.  They were tired of working for other unicorns and decided to strike out on their own.  Their first product was a line of unicorn-themed greeting cards.  The cards were a huge success, and the company grew rapidly.  By 1985, the company had expanded its product line to include unicorn-themed T-shirts, coffee mugs, and stuffed animals.  In 1987, the company went public.  Since then, the Magic Unicorn Corporation has become one of the most successful companies in the world.
   The Magic Unicorn Corporation employs thousands of people around the world.  Most of the employees are unicorns, but the company also employs a small number of humans.  Humans are hired to do jobs that unicorns cannot or will not do.  For example, humans are hired to clean up after unicorns.  Humans are also hired to feed the unicorns

- [完成]

  **Can you help me write a formal email to a potential business partner proposing a joint venture? Your answer:**

  A:
  Dear Sir/Madam,
  I am writing to propose a joint venture between our two companies.  I have attached a proposal for your consideration.  Please let me know if you would like to discuss this further.
   Yours sincerely,

- [完成]

  **using java to sort an unsorted array. Answer:** O(nlogn)

  

  @param arr
  @return
  /**

  \* using java to sort an unsorted array. Answer: O(nlogn)

  \*

  \* @param arr

  \* @return
  */

  public static <T extends Comparable<T>> void sort(T[] arr) {
          if (arr == null || arr.length == 0) {
              return;
          }

  ​		int n = arr.length;
  ​        for (int i = 0; i < n - 1; i++) {
  ​            for (int j = 0; j < n - i - 1; j++) {
  ​                if (arr[j].compareTo(arr[j + 1]) > 0) {
  ​                    T tmp = arr[j];
  ​                    arr[j] = arr[j + 1];
  ​                    arr[j + 1] = tmp;
  ​                }
  ​            }
  ​        }
  ​    }

- [未完成]

  **这是我为我的学生申请浙江大学博士的推荐信：** 尊敬的浙江大学博士研究生招生办公室主任： 您好！  我有幸认识并指导过一位非常优秀的研究生，他就是现在申请浙江大学博士研究生的李某。 我对他的了解和评价如下： 李某于2007年9月进入我的实验室从事博士研究生的学习和科研工作。 在此期间，他刻苦学习、勤奋工作，取得了优异的成绩。 他的学习成绩始终名列前茅，其博士论文被评为校级优秀博士论文。 在科研方面，他参与了多项国家自然科学基金项目的研究工作，发表了多篇高水平的学术论文，其中SCI收录论文5篇，EI收录论文1篇。 他的博士论文已经通过答辩，获得了博士学位。  李某不仅学习成绩优异，而且品德高尚，具有良好的科研素质和创新意识。 他能够独立思考问题，善于总结经验教训，勇于承担责任。 他具有良好的团队合作精神，乐于帮助别人，受到同学和老师的一致好评。  李某具

- [未完成]

  **窗前明月光，疑是地上霜。** 举头望明月，低头思故乡。 这是唐代诗人李白的名句，也是中国古典诗词中最脍炙人口的名句之一。 这首诗抒发了诗人对故乡的思念之情。 作品原文 窗前明月光，疑是地上霜。 举头望明月，低头思故乡。  [1] 作者简介 李白（701～762）字太白，号青莲居士。 祖籍陇西成纪（今甘肃天水附近），先世于隋末流徙西域，李白即生于中亚碎叶（今巴尔喀什湖南面的楚河流域，唐时属安西都护府管辖）。 幼时随父迁居绵州昌隆（今四川江油）青莲乡。 他一生绝大部分在漫游中度过。 公元742年（天宝元年），因道士吴筠的推荐，被召至长安，供奉翰林。 文章风采，名动一时，颇为唐玄宗所赏识。 后因不能见容于权贵，在京仅三年，就弃官而去，仍然继续他那飘荡四方的流浪生活。 公元756年，即安史之乱发生的第二年，他感愤时艰，曾

- [未完成]

  **你阅读过李白的所有诗歌。李白的《将进酒》的原文是**:君不见黄河之水天上来,奔流到海不复回。 君不见高堂明镜悲白发,朝如青丝暮成雪。 人生得意须尽欢,莫使金樽空对月。 天生我材必有用,千金散尽还复来。 烹羊宰牛且为乐,会须一饮三百杯。 岑夫子,丹丘生,将进酒,君莫停。 与君歌一曲,请君为我侧耳听。 钟鼓馔玉何足贵,但愿长醉不复醒。 古来圣贤皆寂寞,惟有饮者留其名。 陈王昔时宴平乐,斗酒十千恣欢谑。 主人何为言少钱,径须沽取对君酌。 五花马,千金裘,呼儿将出换美酒,与尔同销万古愁。 唐代大诗人李白(701-762),字太白,号青莲居士。 祖籍陇西成纪(今甘肃天水),隋末其先人流寓西域碎叶(唐时属安西都护府,在今吉尔吉斯斯坦北部托克马克附近),李白即生于碎叶。 幼时随父迁居绵州昌隆(今四川江油)青莲乡

- [未完成]

  **You are now a doctor. Here are some tips for a healthy life. 1.**  Don't smoke.  2.  Eat a balanced diet.  3.  Exercise regularly.  4.  Drink alcohol only in moderation.  5.  Take care of your teeth.  6.  Get enough sleep.  7.  See your doctor for regular checkups.  8.  Learn to manage stress.  9.  Stay up-to-date on vaccinations.  10.  Wash your hands often.  11.  Avoid secondhand smoke.  12.  Use sunscreen and wear sunglasses.  13.  Know your family medical history.  14.  Be aware of the signs of depression.  15.  Protect your hearing.  16.  Practice safe sex.  17.  If you drink coffee, do so in moderation.  18.  Limit your intake of caffeine.  19.  Don't drive when you're tired.  20.  Don't text and drive.  21.  Don't drink and drive.  22.  Don't ride a motorcycle without a helmet.  23.  Don't swim alone.  24.  Don't dive into water where you can't see the bottom.  25.  Don't swim if you've been drinking alcohol.  26.  Don't swim if you're sick.  27.  Don't swim if you have open sores on your body.  28.  Don't swim if there are strong currents or undertows.  29.  Don't swim in polluted water.  30.  Don't swim if there are snakes or alligators nearby.  31.  Don't swim if there are jellyfish in the water.  32.  Don't swim if there are sharks in the water.  33.  Don't swim if there are riptides in the water.  34.  Don't

- [未完成]

  **你非常了解中国的大学。请介绍一下浙江大学。** 浙江大学是中国著名的高等学府之一,它的前身是创建于1897年的求是书院,1928年改为国立浙江大学,1952年定名为浙江大学。 浙江大学是首批进入国家"211工程"和"985工程"重点建设的高校之一。 浙江大学有着悠久的历史和深厚的文化底蕴,在中国近现代教育史上占有重要地位。 浙江大学的前身求是书院创建于1897年,是中国人最早自己创办的新式高等学府之一。 1901年,求是书院更名为浙江大学堂。 1928年,浙江大学正式定名为国立浙江大学。 1952年,浙江大学成为全国首批重点大学。 1958年,浙江大学成为全国首批博士、硕士学位授予单位。 1998年,浙江大学成为首批进入国家"211工程"重点建设的高校。 2001年,浙江大学成为首批进入国家"985工程"重点建设的高校。 2006年,浙江大学成为首批进入国家"111计划"重点建设的高校。 2017年,浙江大学成为首批进入国家"双一流"世界一流大学和一流学科建设高校。 浙江大学是一所研究型、综合性、

- [未完成]

  **你对中国的大学了解的非常多。请介绍一下浙江大学。答案：**浙江大学是中国著名的高等学府之一，坐落在美丽的钱塘江畔，有着悠久的历史和深厚的文化底蕴。 浙江大学创办于1897年，是中国人最早自己创办的新式高等学府之一，也是中国近代高等教育的发源地之一。 浙江大学是首批进入国家“211工程”和“985工程”建设行列的高校之一，也是国家“111计划”、“珠峰计划”、“卓越工程师教育培养计划”、“卓越医生教育培养计划”、“世界一流大学和一流学科”建设高校之一。 浙江大学现有三个校区，分别位于杭州、嘉兴和海宁，占地面积3000余亩，建筑面积100余万平方米，图书馆藏书600余万册。 浙江大学现有教职员工5000余人，其中专任教师2000余人，包括中国科学院院士14人、中国工程院院士11人、国家杰出青年科学基金获得者30人、长江学者特聘教授30人、国家“千人计划”入选者30人、国家“青年千人计划”入选者30人、教育部“新世纪优秀人才支持

- [未完成]

  **You are very familiar with China. Next, please introduce the attractions, status, and cuisine of Beijing. The capital of China is** Beijing.  It is the political, economic, cultural, and educational center of China.  Beijing has a history of more than 3,000 years.  It was first built in 1045 B.C.  by King Wu of the Zhou Dynasty (1045-256 B.C.).  In 1949, Beijing became the capital of the People's Republic of China.  Beijing covers an area of 16,800 square kilometers.  It is divided into 14 districts and 2 counties.  There are about 17 million people living in Beijing.  Beijing is located in the northern part of the North China Plain.  It is surrounded by mountains in the north, west, and south.  To the east is the Bohai Sea.  Beijing has a monsoon-influenced humid continental climate.  The average annual temperature is 11.8 degrees Celsius.  The best time to visit Beijing is spring and autumn.  Beijing has many famous attractions.  The Forbidden City, Tiananmen Square, the Great Wall, the Summer Palace, the Temple of Heaven, the Lama Temple, the Ming Tombs, the Fragrant Hills, the Yungang Grottoes, the Badaling Great Wall, the Mutianyu Great Wall, the Juyongguan Pass, the Longqing Gorge, the Chengde Mountain Resort, the Olympic Green, the Bird's Nest, the Water Cube, the National Grand Theater, the National Library, the National Art Museum, the National Museum of China, the National Center for the Performing Arts, the Capital Museum, the Military Museum, the Beijing Zoo, the Beijing Aquarium, the Beijing Botanical Garden, the Beijing Planetarium, the Beijing Natural History Museum, the Beijing Ancient Observatory, the Beihai Park, the Jingshan Park, the Drum Tower, the Bell Tower, the Confucius Temple, the White Pagoda Temple, the Guanghua Temple, the Fayuan Temple, the Imperial Ancestral Temple, the Zhongshan Park, the Wangfujing Street, the Q

- [删重]

  **Question: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
  Answer: Roger started with 5 balls. 2 cans of 3 each is 6 tennis balls. 5 + 6 = 11. The answer is 11.
  Question: The cafeteria had 23 apples. lf they used 20 to make lunch and bought 6 more, how many apples do they have?
  Answer: Cafeteria started with 23 apples.**  20 apples were used to make lunch.  6 more apples were bought.  23 - 20 = 3.  3 + 6 = 9.  The answer is 9.
  Question: If you have 4 marbles and I have 3 marbles, how many marbles do we have together?
  Answer: 4 + 3 = 7.  The answer is 7.

- [删重]

  **我爱你的英文是什么？** 我爱你的英文是"i love you"。

  

### 2. 指令微调





## 预训练

### 1. 语料



### 2. 训练



### 3. 推理



## 指令微调

### 1. 语料



### 2. 训练



### 3. 推理





## 预训练权重恢复

> 由于meta官方没有完全开放出LLaMA的权重，因此我们将CaMA的权重与LLaMA权重进行作差，上传到了[此处](https://huggingface.co/zjunlp/CaMA-13B)。为了还原得到完整的CaMA权重，请按照下述步骤进行。

### 1. 下载LlaMA 13B官方模型

请点击[此处](https://forms.gle/jk851eBVbX1m5TAv5)向`meta`申请`LLaMA`的官方预训练权重。此处我们使用的是`13B`规格的模型，因此仅需下载`13B`版本即可。下载完成后的文件目录如下：

```shell
|-- 13B
|	|-- checklist.chk
|	|-- consolidated.00.pth
|	|-- consolidated.01.pth
|	|-- params.json
|-- llama.sh
|-- tokenizer.model
|-- tokenizer_checklist.chk
```

## 2. 使用Huggingface提供的转换脚本

使用Huggingface提供的脚本文件，对原始的`LLaMA-13B`转换为Huggingface的格式，具体的脚本文件在[此处](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py)。下面是运行的命令（假设下载的原始文件位于`./`下，希望转换后的路径为`./converted`）：

```shell
python convert_llama_weights_to_hf.py --input_dir ./ --model_size 13B --output_dir ./converted
```

## 3. 使用脚本复原CaMA 13B

最后使用我们提供的脚本，位于`./tools/weight_diff.py`，执行下面的命令，将得到完整的`CaMA`权重：

```shell
python tools/weight_diff.py recover --path_raw ./converted --path_diff zjunlp/CaMA-13B --path_tuned ./recover
```

最后完整的权重被保存在recover文件夹中。