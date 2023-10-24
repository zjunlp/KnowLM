:speaking_head: \[ [中文](./README_ZH.md) | **English** \]

<p align="center">
    <br>
    <img src="https://github.com/zjunlp/KnowLM/blob/main/assets/KnowLM.png?raw=true" width="400" height="120"/>
    <br>
</p>

# Knowledgeable Large Language Model Framework

<details>
  <summary><b>Project Background</b></summary>
With the rapid development of deep learning technology, large language models such as ChatGPT have made substantial strides in the realm of natural language processing. However, these expansive models still encounter several challenges in acquiring and comprehending knowledge, including the difficulty of updating knowledge and potential knowledge discrepancies and biases, collectively known as <b>knowledge fallacies</b>. The KnowLM project endeavors to tackle these issues by launching an open-source large-scale knowledgable language model framework and releasing corresponding models. 

The project's `initial phase` introduced a knowledge extraction LLM based on LLaMA, dubbed **ZhiXi** (**智析**, which means intelligent analysis of data for knowledge extraction). To integrate the capacity of Chinese understanding into the language models without compromising their inherent knowledge, we firstly <b>(1) use Chinese corpora for the full-scale pre-training with LLaMA (13B), augment the language model's understanding of Chinese and improve its knowledge richness while retaining its original English and code capacities;</b> Then <b>(2) we fine-tune the model obtained from the first step with an instruction dataset, thus bolstering the language model's understanding of human instructions for knowledge extraction.</b>
- ❗Please note that this project is still undergoing optimization, and the model weights will be regularly updated to support new features and models!

</details>


**The features of this project are as follows:**

- Centered on knowledge and large models, a **full-scale pre-training** of the large model, such as LLaMA, is conducted using the built Chinese&English pre-training corpus.
- Based on the technology of **KG2Instructions**, the knowledge extraction tasks, including NER, RE, and IE, are optimized and can be completed using human instructions.
- Using the built Chinese instruction dataset (approximately 1400K), LoRA fine-tuning is used to enhance the model's understanding of human instructions.
- The weights of the pre-training model and LoRA's instruction fine-tuning are open-sourced.
- The **full-scale pre-training code** (providing conversion, construction, and loading of large corpora) and **LoRA instruction fine-tuning code** are open-sourced (support multi-machine multi-GPU).


All weights and datasets have been uploaded to HuggingFace🤗. Click [here](#1-1) to get started right away!

❗**If you encounter any issues during the installation or using of KnowLM, please check [FAQ](https://github.com/zjunlp/KnowLM#6) or promptly submit an [issue](https://github.com/zjunlp/KnowLM/issues), and we will assist you with resolving the problem!**

| Category | Base   | Name                     | Version | Download Link                                                     | Note     |
| -------- | ------ | ------------------------- | ---- | ------------------------------------------------------------ | -------- |
| Base Model | LlaMA1 | KnowLM-13B-Base           | V1.0 | [HuggingFace](https://huggingface.co/zjunlp/knowlm-13b-base-v1.0) | Base Model |
| Dialogue Model | LlaMA1 | KnowLM-13B-ZhiXi          | V1.0 | [HuggingFace](https://huggingface.co/zjunlp/knowlm-13b-zhixi) | Information Extraction Model |
| Dialogue Model | LlaMA1 | KnowLM-13B-IE             | V1.0 | [HuggingFace](https://huggingface.co/zjunlp/knowlm-13b-ie)  | Information Extraction Model |
| Base Model | LlaMA2 | KnowLM-7B-Base            | V1.0 | Coming soon                                                     | Base Model |
| Dialogue Model | LlaMA2 | KnowLM-7B-IE              | V1.0 | Coming soon                                                     | Information Extraction Model |
| Dialogue Model | LlaMA2 | KnowLM-7B-Ocean(OceanGPT) | V1.0 | [HuggingFace]( ) | Ocean Model |
| Dialogue Model | LlaMA2 | KnowLM-13B                | V1.0 | Training                                                       | Base Model |

| Instruction Dataset Name                        | Number    | Download Link                                                    | Is it used by ZhiXi | Note                           |
| ------------------------------- | ------- | ------------------------------------------------------------ | ------------ | ------------------------------ |
| KnowLM-CR (CoT&Reasoning, Chinese and English) | 202,333 | [Google Drive](https://drive.google.com/drive/folders/1iJgksjOStk0m9GM0RP9jB6KdNWfJ62Xe?usp=sharing) <br/> [HuggingFace](https://huggingface.co/datasets/zjunlp/KnowLM-CR)| Yes           |                              |
| KnowLM-IE (Information Extraction, Chinese)         | 281,860 | [Google Drive](https://drive.google.com/file/d/1WQVD_99_4XoUcoRDWRibZfO5jJdhjTQ1/view?usp=sharing) <br/> [HuggingFace](https://huggingface.co/datasets/zjunlp/KnowLM-IE) | Yes           | Due to using distant supervision, there exists noise. |
| KnowLM-Tool (Tool Learning，English)     | 38,241  | [Google Drive](https://drive.google.com/file/d/1PyzXXv_pr2T-FysnCumWTDzFNCvtLDv2/view?usp=sharing) <br/> [HuggingFace](https://huggingface.co/datasets/zjunlp/KnowLM-Tool) | No           | It will be used in the next version.                             |

**Data description**: 1. Other data sources for information extraction come from `CoNLL`, `ACE`, `casis`, `DuEE`, `People Daily`, `DuIE`, etc. 2. The `KnowLM-Tool` dataset comes from the paper "[Making Language Models Better Tool Learners with Execution Feedback](https://arxiv.org/abs/2305.13068)" and the [gitHub](https://github.com/zjunlp/trice) can be found here. 3. The `KnowLM-IE` dataset comes from the paper "[InstructIE: A Chinese Instruction-based Information Extraction Dataset](https://arxiv.org/abs/2305.11527)" and the [gitHub](https://github.com/zjunlp/DeepKE/tree/main/example/llm/InstructKGC) can be found here.

## 📬 NEWS
- \[**August 2023**\] The full parameters have been released (omitting the parameter consolidation process).
- \[**July 2023**\] The instruction dataset has been released.
- \[**July 2023**\] Support instruction fine-tuning and vllm for `LLaMA-2`
- \[**June 2023**\] The project name has been changed from `CaMA` to `KnowLM`.
- \[**June 2023**\] Release the first version of pre-trained weights and the LoRA weights.

## 📍 What's the KnowLM?
<p align="center">
    <br>
    <img src="https://github.com/zjunlp/KnowLM/blob/main/assets/KnowLM-overview.png?raw=true" width="920" height="400"/>
    <br>
</p>

This is an overview of the `KnowLM`, which mainly consists of three technical features:

**Knowledge Prompting**: It generates knowledge prompts based on structured data such as knowledge graphs and utilizes knowledge augmentation constraints to address *knowledge extraction and reasoning* issues.

**Knowledge Editing**: It aligns outdated, incorrect, and biased knowledge within large models using knowledge editing techniques to tackle *knowledge fallacy* problems (**[English Tutorial](./pdf/Knowledge_Editing.pdf)**).

**Knowledge Interaction**: It enables dynamic knowledge interaction and feedback to achieve tool-based learning and multi-agent collaboration, resolving the problem of *embodiment cognition* in LLMs (**[English Tutorial](./pdf/Knowledge_Interaction.pdf)**).

The tools corresponding to these three technologies are [EasyInstruct](https://github.com/zjunlp/easyinstruct), [EasyEdit](https://github.com/zjunlp/easyedit), and EasyAgent (under development). We will soon provide use cases for knowledge prompting and knowledge editing based on the `KnowLM`framework.

## 🗂️ Contents

- [🚴Quick Start](#1-quick-start)
  - [🛠️Environment Configuration](#%EF%B8%8F11-environment-configuration)
  - [💻Model Usage Guide](#12-model-usage-guide)
  - [🎯Information Extraction prompt](#13-information-extraction-prompt)
  - [🐐Llama.cpp](#14-llamacpp)
  - [🖊️Model Editing](#%EF%B8%8F15-model-editing)

- [🌰Cases](#2-cases)
  - [🌰Pretraining Cases](#21-pretraining-cases)
  - [🌰Information Extraction Cases](#22-information-extraction-cases)
  - [🌰General Ability Cases](#23-general-ablities-cases)
  - [🌰Model Editing case](#24-model-editing-cases)
- [🥊Training Details](#3-training-details)
  - [🧾Pertraining data and Pretraining scripts](#31-dataset-construction-pretraining)
  - [🧾Instruction data and Instruction-tuning scripts](#32-training-process-pretraining)
- [🔴Limitations](#4-limitations)
- [🕐TODO List](#5-todo-list)
- [❓FAQ](#6-faq)
- [👋Acknowledgments/Contributors/Citations](#7-others)


<h2 id="1">🚴1. Quick Start</h2>

<h3 id="1-1">🛠️1.1 Environment Configuration</h3>

*KnowLM* supports both **manual** and **docker image** environment configuration, you can choose the appropriate way to build.
#### 🔧Manual Environment Configuration
```shell
git clone https://github.com/zjunlp/KnowLM.git
cd KnowLM
conda create -n knowlm python=3.9 -y
conda activate knowlm
pip install torch==1.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```
#### 🐳Building With Docker Images
```shell
docker pull zjunlp/knowlm:v.1
docker run -it zjunlp/knowlm:v.1 /bin/bash
```
<h3 id="1-2">💻1.2 Model Usage Guide</h3>

**1. Reproduce the results in Section 2**

> The cases in **[Section 2](#2-cases)** were all run on V100. If running on other devices, the results may vary. Please run multiple times or change the decoding parameters. **We derived `knowlm-13b-zhixi` and `knowlm-13b-ie` through training using LoRA, building upon the foundation of `knowlm-13b-base`. These models, `knowlm-13b-zhixi` and `knowlm-13b-ie`, are the result of merging the trained LoRA weights with the existing `knowlm-13b-base` model parameters.**

1. If you want to reproduce the results in `section 2.1`(**[pretraining cases](#21-pretraining-cases)**), please run the following command:

   ```shell
   python examples/generate_finetune.py --base_model zjunlp/knowlm-13b-base-v1.0
   ```

   The result in section `2.1` can be obtained.

2. If you want to reproduce the results in `section 2.2`(**[information extraction cases](#22-information-extraction-cases)**), please run the following command:

   ```shell
   python examples/generate_lora.py --base_model zjunlp/knowlm-13b-zhixi --run_ie_cases
   ```

   The result in section `2.2` can be obtained.

3. If you want to reproduce the results in `section 2.3`(**[general ablities cases](#23-general-ablities-cases)**), please run the following command:

   ```shell
   python examples/generate_lora.py --base_model zjunlp/knowlm-13b-zhixi --run_general_cases
   ```

   The result in section `2.3` can be obtained.


**2. Usage of Pretraining Model**

We offer two methods: the first one is **command-line interaction**, and the second one is **web-based interaction**, which provides greater flexibility.

1. Use the following command to enter **command-line interaction**:

   ```shell
   python examples/generate_finetune.py --base_model zjunlp/knowlm-13b-base-v1.0 --interactive
   ```

   The disadvantage is the inability to dynamically change decoding parameters.

2. Use the following command to enter **web-based interaction**:

   ```shell
   python examples/generate_finetune_web.py --base_model zjunlp/knowlm-13b-base-v1.0
   ```
   Here is a screenshot of the web-based interaction:
   <p align="center" width="100%">
   <a href="" target="_blank"><img src="./assets/finetune_web.jpg" alt="finetune-web" style="width: 100%; min-width: 100px; display: block; margin: auto;"></a>
   </p>

**3. Usage of Instruction tuning Model**

Here, we provide a web-based interaction method. Use the following command to access the web:

```shell
python examples/generate_lora_web.py --base_model zjunlp/knowlm-13b-zhixi
```

Here is a screenshot of the web-based interaction:
<p align="center" width="100%">
<a href="" target="_blank"><img src="./assets/lora_web.png" alt="finetune-web" style="width: 100%; min-width: 100px; display: block; margin: auto;"></a>
</p>

The `instruction` is a required parameter, while `input` is an optional parameter. For general tasks (such as the examples provided in section `1.3`), you can directly enter the input in the `instruction` field. For information extraction tasks (as shown in the example in section `1.2`), please enter the instruction in the `instruction` field and the sentence to be extracted in the `input` field. We provide an information extraction prompt in section `2.5`. 

If you want to perform batch testing, please modify the `examples/generate_lora.py` file and update the examples and hyperparameters in the variable `cases`.

According to different task requirements, we have the following suggestions for adjusting decoding strategies and their associated hyperparameters:

1. If you want more diverse and creative outputs, consider using top-k or top-p (nucleus) sampling with a relatively higher `top_k` or `top_p`, and possibly a higher `temperature`.
2. If you want more focused and high-quality outputs (e.g., information extraction), consider using beam search with a moderate `num_beam`, or top-k or top-p sampling with a lower `top_k` or `top_p`, and a lower `temperature`.
3. Remember to experiment and fine-tune. Depending on your use case, it may be beneficial to iterate and experiment with different strategies and hyperparameters to find the optimal combination.

**4. vLLM API server**

We integrate [vLLM](https://github.com/vllm-project/vllm) for accelerating LLM inference and providing efficient API service. Use the following command to launch vLLM API server at `http://localhost:8090`.

```shell
max_num_batched_tokens=8000

CUDA_VISIBLE_DEVICES=1,2 python inference/launch_vllm.py \
    --port 8090 \
    --model data/zhixi-13B \
    --use-np-weights \
    --max-num-batched-tokens $max_num_batched_tokens \
    --dtype half \
    --tensor-parallel-size 2
```

Query the service using POST request:

```shell
curl -X POST "http://127.0.0.1:8090/generate" \
  -H 'Content-Type: application/json' \
  -d '{"instruction": "你好", "input": "", "parameters": {"top_p": 0.7, "max_tokens": 256}}'
```

You could get the following response:

```shell
{
  "generated_text":"你好，很高兴见到你。我是一个人工智能助手，可以帮助你解决问题和提供信息。有什么我可以帮助你的吗？</s>",
  "num_output_tokens_cf":65,
  "error":null
}
```

<h3 id="1-3">🎯1.3 Information Extraction Prompt</h3>

For information extraction tasks such as named entity recognition (NER), event extraction (EE), and relation extraction (RE), we provide some prompts for ease of use. You can refer to this [link](./examples/ie_prompt.py) for examples. Of course, you can also try using your own prompts.

Here is a [case](https://github.com/zjunlp/DeepKE/blob/main/example/llm/InstructKGC/README.md) where `knowlm-13b-zhixi` is used to accomplish the instruction-based knowledge graph construction task in CCKS2023.

<h3 id="1-4">🐐1.4 LlaMA.cpp</h3>

If you find yourself lacking sufficient GPU computing resources, you have the option to carry out quantization using [llama.cpp](https://github.com/ggerganov/llama.cpp). This is possible because llama.cpp shares the same architecture as KnowLM. Once you have set up your environment, you can download our model to a designated path using the following command:
```bash
python tools/download.py --specify --download_path ./your/path --repo_name zjunlp/knowlm-13b-zhixi
```
Next, just substitute the model path at this [location](https://github.com/ggerganov/llama.cpp#prepare-data--run) with the downloaded one. When executing it in practice, please remember to adjust the model path within this [script](https://github.com/ggerganov/llama.cpp/blob/master/examples/alpaca.sh) accordingly.

<h3 id="1-5">🖊️1.5 Model Editing</h3>

Although large language models perform exceptionally well in many tasks, they can still provide incorrect answers. Moreover, as time passes, knowledge that was once accurate may become outdated. This necessitates that we adjust the model's responses to meet our expectations through model editing.

In model editing, we utilized EasyEdit as our editing tool (details can be found at https://github.com/zjunlp/EasyEdit). EasyEdit is a highly integrated model editing tool. All you need to do is define your editor in just three lines of code, similar to how you would in hugging face.
```shell
from easyeditor import MENDHyperParams
hparams = MENDHyperParams.from_hparams('./hparams/MEND/gpt2-xl')
editor = BaseEditor.from_hparams(hparams)
```
The code above demonstrates the editor definition for editing the gpt2-xl model using the MEND method. The next step is to prepare the editing data and the test data.

```shell
metrics, edited_model, _ = editor.edit(
    prompts=prompts,
    ground_truth=ground_truth,
    target_new=target_new,
    locality_inputs=locality_inputs,
    keep_original_weight=True
)
```
With the provided code, you can complete the editing of the model. The edited model is stored in "edit_model," and the corresponding evaluation metrics are saved in "metrics."

<h2 id="2">🌰2. Cases</h2>

<h3 id="2-1">🌰2.1 Pretraining Cases</h3>

Our pre-trained model has demonstrated certain abilities in instruction following, coding, reasoning, as well as some translation capabilities, without any fine-tuning using instructions. Additionally, it has acquired new knowledge. Below are some of our sample cases. If you wish to reproduce our examples and view detailed decoding configuration, please first [set up the environment](#1-1), then follow the steps outlined [here](#1-2).

> In the following cases, text in **bold** represents the prompt, while non-bold text represents the model's output.
>
> Due to the maximum inference length set to 512, our cases fall into three situations:
>
> 1. Completed output. The model generates the termination token EOS and completes the output. We mark this with :white_check_mark:.
> 2. Incomplete output. The output is cut off due to the maximum inference length. We mark this with :eight_spoked_asterisk:.
> 3. Repeated output. We remove repeated content manually and mark it with :arrow_left:.

<details>
  <summary><b>Translation</b></summary>


  - <b>你很擅长将中文翻译成英文。将下面的句子翻译成英文：我们今天准备去西安看兵马俑。答案：</b>We are going to see the Terracotta Warriors in Xi'an today.:white_check_mark:

- **You are good at translating English into Chinese. Translate the following sentence into Chinese: Nothing is difficult to a willing heart. Answer:** 什么都不难,只要心甘情愿。:white_check_mark:
  </details>
  

<details>
  <summary><b>Knowledge</b></summary>


- **你阅读过李白的所有诗歌。李白的《将进酒》的原文是**:君不见黄河之水天上来,奔流到海不复回。 君不见高堂明镜悲白发,朝如青丝暮成雪。 人生得意须尽欢,莫使金樽空对月。 天生我材必有用,千金散尽还复来。 烹羊宰牛且为乐,会须一饮三百杯。 岑夫子,丹丘生,将进酒,君莫停。 与君歌一曲,请君为我侧耳听。 钟鼓馔玉何足贵,但愿长醉不复醒。 古来圣贤皆寂寞,惟有饮者留其名。 陈王昔时宴平乐,斗酒十千恣欢谑。 主人何为言少钱,径须沽取对君酌。 五花马,千金裘,呼儿将出换美酒,与尔同销万古愁。 唐代大诗人李白(701-762),字太白,号青莲居士。 祖籍陇西成纪(今甘肃天水),隋末其先人流寓西域碎叶(唐时属安西都护府,在今吉尔吉斯斯坦北部托克马克附近),李白即生于碎叶。 幼时随父迁居绵州昌隆(今四川江油)青莲乡:eight_spoked_asterisk:
- **床前明月光，疑是地上霜。** 举头望明月，低头思故乡。 这首诗出自唐代诗人李白的《静夜思》。 作品原文 床前明月光，疑是地上霜。 举头望明月，低头思故乡。  [1] 作者简介 李白（701～762）字太白，号青莲居士。 祖籍陇西成纪（今甘肃天水附近），先世于隋末流徙西域，李白即生于中亚碎叶（今巴尔喀什湖南面的楚河流域，唐时属安西都护府管辖）。 幼时随父迁居绵州昌隆（今四川江油）青莲乡。 他一生绝大部分在漫游中度过。 公元742年（天宝元年），因道士吴筠的推荐，被召至长安，供奉翰林。 文章风采，名动一时，颇为唐玄宗所赏识。 后因不能见容于权贵，在京仅三年，就弃官而去，仍然继续他那飘荡四方的流浪生活。 公元756年，即安史之乱发生的第二年，他感愤时艰，曾参加了永王李璘的幕府。 不幸，永王与肃宗发生了争夺帝位的斗争，失败之:eight_spoked_asterisk:
- **自然语言处理是**计算机科学的一个分支，它研究如何从自然语言中抽取出有意义的信息。 自然语言处理包括词法分析、句法分析、语义分析、语音识别和机器翻译等。 自然语言处理的目标是使计算机能够像人类一样理解自然语言。:white_check_mark:
</details>


<details>
  <summary><b>Instruction Following</b></summary>


- <b>你非常了解一些健康生活的习惯，请列举几个健康生活的建议：</b>1.每天坚持锻炼30分钟以上。  2.不吸烟，不酗酒。  3.少吃高脂肪食物。  4.多吃蔬菜和水果。  5.保证充足的睡眠。  6.保持良好的心情。  7.定期体检。  8.养成良好的卫生习惯。:white_check_mark:

- **我爱你的英文是什么？** 我爱你的英文是"i love you"。:arrow_left:

- **你对中国的大学了解的非常多。请评价一下浙江大学是一所怎样的学校。** 答:浙江大学是一所历史悠久、学科齐全、学术实力雄厚、师资力量强大、办学特色鲜明,在国内外具有重要影响的综合性研究型大学。 浙江大学是首批进入国家"211工程"和"985工程"建设的高校之一,也是国家"111计划"和"珠峰计划"重点建设的名牌大学。 浙江大学的前身是创建于1897年的求是书院,是中国人最早自己创办的新式高等学府之一。 1928年,求是书院改制为国立浙江大学。 1952年,浙江大学成为新中国创办的第一批全国重点大学。 1998年,浙江大学与杭州大学、浙江农业大学、浙江医科大学、浙江师范大学、浙江师范大学医学院合并组建新的浙江大学。 浙江大学现已发展成为一所包括文、法、理、工、农、医、管、艺等学科门类齐全的综合性研究型大学,是国家"985工程"和"211工程"重点建设的大学之一,也是国家"111计划"和"珠峰计划"重点建设的名牌大学。:white_check_mark:

- **你非常了解中国的大学。请介绍一下浙江大学。** 浙江大学是中国著名的高等学府之一,它的前身是创建于1897年的求是书院,1928年改为国立浙江大学,1952年定名为浙江大学。 浙江大学是首批进入国家"211工程"和"985工程"重点建设的高校之一。 浙江大学有着悠久的历史和深厚的文化底蕴,在中国近现代教育史上占有重要地位。 浙江大学的前身求是书院创建于1897年,是中国人最早自己创办的新式高等学府之一。 1901年,求是书院更名为浙江大学堂。 1928年,浙江大学正式定名为国立浙江大学。 1952年,浙江大学成为全国首批重点大学。 1958年,浙江大学成为全国首批博士、硕士学位授予单位。 1998年,浙江大学成为首批进入国家"211工程"重点建设的高校。 2001年,浙江大学成为首批进入国家"985工程"重点建设的高校。 2006年,浙江大学成为首批进入国家"111计划"重点建设的高校。 2017年,浙江大学成为首批进入国家"双一流"世界一流大学和一流学科建设高校。 浙江大学是一所研究型、综合性、:eight_spoked_asterisk:

- **Can you help me write a formal email to a potential business partner proposing a joint venture? Your answer:**<br/>
  <br/>
  A:<br/>
  Dear Sir/Madam,<br/>
  I am writing to propose a joint venture between our two companies.  I have attached a proposal for your consideration.  Please let me know if you would like to discuss this further.<br/>
   Yours sincerely,:white_check_mark:

- **You are now a doctor. Here are some tips for a healthy life. 1.**  Don't smoke.  2.  Eat a balanced diet.  3.  Exercise regularly.  4.  Drink alcohol only in moderation.  5.  Take care of your teeth.  6.  Get enough sleep.  7.  See your doctor for regular checkups.  8.  Learn to manage stress.  9.  Stay up-to-date on vaccinations.  10.  Wash your hands often.  11.  Avoid secondhand smoke.  12.  Use sunscreen and wear sunglasses.  13.  Know your family medical history.  14.  Be aware of the signs of depression.  15.  Protect your hearing.  16.  Practice safe sex.  17.  If you drink coffee, do so in moderation.  18.  Limit your intake of caffeine.  19.  Don't drive when you're tired.  20.  Don't text and drive.  21.  Don't drink and drive.  22.  Don't ride a motorcycle without a helmet.  23.  Don't swim alone.  24.  Don't dive into water where you can't see the bottom.  25.  Don't swim if you've been drinking alcohol.  26.  Don't swim if you're sick.  27.  Don't swim if you have open sores on your body.  28.  Don't swim if there are strong currents or undertows.  29.  Don't swim in polluted water.  30.  Don't swim if there are snakes or alligators nearby.  31.  Don't swim if there are jellyfish in the water.  32.  Don't swim if there are sharks in the water.  33.  Don't swim if there are riptides in the water.  34.  Don't:eight_spoked_asterisk:
</details>


<details>
  <summary><b>Coding</b></summary>


- **using java to sort an unsorted array. Answer:** O(nlogn)

  

  @param arr
  @return
  /\*\*

  \* using java to sort an unsorted array. Answer: O(nlogn)

  \*

  \* @param arr

  \* @return
  \*/

  public static <T extends Comparable\<T\>\> void sort(T\[\] arr) {<br/>
  &nbsp;&nbsp;&nbsp;&nbsp;if (arr == null || arr.length == 0) {<br/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return;<br/>
  &nbsp;&nbsp;&nbsp;&nbsp;}<br/>
  &nbsp;&nbsp;&nbsp;&nbsp;int n = arr.length;<br/>
  &nbsp;&nbsp;&nbsp;&nbsp;for (int i = 0; i < n - 1; i++) {<br/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;for (int j = 0; j < n - i - 1; j++) {<br/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if (arr[j].compareTo(arr[j + 1]) > 0) {<br/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;T tmp = arr[j];<br/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;arr[j] = arr[j + 1];<br/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;arr[j + 1] = tmp;<br/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;}<br/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;}<br/>
  &nbsp;&nbsp;&nbsp;&nbsp;}<br/>
  }:white_check_mark:

- **use python to write the bubble sort algorithm.**

  def bubble_sort(arr):<br/>
  &nbsp;&nbsp;&nbsp;&nbsp;"""Bubble sort algorithm.<br/>
  &nbsp;&nbsp;&nbsp;&nbsp;""" for i in range(len(arr) - 1, 0, -1):<br/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;for j in range(len(arr) - 1, 0, -1):<br/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if arr[j] > arr[j + 1]:<br/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;arr[j], arr[j + 1] = arr[j + 1], arr[j]<br/>
  &nbsp;&nbsp;&nbsp;&nbsp;return arr:white_check_mark:
  

</details>

<details>
  <summary><b>Generate long text in Chinese</b></summary>



- **这是我为我的学生申请浙江大学博士的推荐信：** 尊敬的浙江大学博士研究生招生办公室主任： 您好！  我有幸认识并指导过一位非常优秀的研究生，他就是现在申请浙江大学博士研究生的李某。 我对他的了解和评价如下： 李某于2007年9月进入我的实验室从事博士研究生的学习和科研工作。 在此期间，他刻苦学习、勤奋工作，取得了优异的成绩。 他的学习成绩始终名列前茅，其博士论文被评为校级优秀博士论文。 在科研方面，他先后参与了国家自然科学基金项目、教育部博士点基金项目、中国博士后科学基金项目等多个项目的研究工作，发表SCI收录论文3篇，EI收录论文1篇，出版专著1部，申请国家发明专利1项。 他还积极参加社会实践活动，曾获得全国大学生“挑战杯”创业计划竞赛二等奖。  李某不仅学习成绩优异，而且品德高尚，具有良好的科研素质和创新精神。 他能够独立思考问题，善于总结经验教训，勇于承担责任。 :eight_spoked_asterisk:
  </details>
  
<details>
  <summary><b>Generate long text in English</b></summary>


- **Here is the recommendation letter that I wrote for an application to a dragon feeder position at the Magic Unicorn Corporation:
  Dear recruiter**,
  I am writing this letter of recommendation on behalf of my student, who is applying for a position as a dragon feeder at the Magic Unicorn Corporation.  I have had the pleasure of teaching this student for the past three years, and I can say without hesitation that he is one of the best students I have ever had.  He is hard-working, diligent, and intelligent.  He is also a very good dragon feeder.  In fact, he is the best dragon feeder I have ever had in my class.  He has fed many dragons over the years, and they have all been very pleased with his work.  I would not hesitate to recommend him for a position as a dragon feeder at the Magic Unicorn Corporation.
   Thank you for your consideration.
   Sincerely,
  Professor Snape
  P.S.  Please don't tell anyone that I wrote this letter.  I don't want to get into any trouble with the Ministry of Magic.

   \# Chapter 1

  \# THE MAGIC UNICORN CORPORATION

  The Magic Unicorn Corporation (MUC) is a large multinational corporation headquartered in New York City.  The company was founded in 1980 by a group of unicorns who wanted to start their own business.  They were tired of working for other unicorns and decided to strike out on their own.  Their first product was a line of unicorn-themed greeting cards.  The cards were a huge success, and the company grew rapidly.  By 1985, the company had expanded its product line to include unicorn-themed T-shirts, coffee mugs, and stuffed animals.  In 1987, the company went public.  Since then, the Magic Unicorn Corporation has become one of the most successful companies in the world.<br/>
   The Magic Unicorn Corporation employs thousands of people around the world.  Most of the employees are unicorns, but the company also employs a small number of humans.  Humans are hired to do jobs that unicorns cannot or will not do.  For example, humans are hired to clean up after unicorns.  Humans are also hired to feed the unicorns:eight_spoked_asterisk:

- **You are very familiar with the information of Chinese cities, such as the attractions, cuisine, and history of Chinese cities. Please introduce the city of Hangzhou. Hangzhou** is the capital of Zhejiang Province in eastern China.  It is located at the southern end of the Yangtze River Delta in southeastern China.  It is one of the most famous tourist cities in China.  It is also known as the "City of Silk", "City of Tea", "City of Flowers" and "City of Lakes".  It has a population of 6.5 million people.  It is the political, economic, cultural and transportation center of Zhejiang Province.  It is also an important industrial base in China.  In addition, Hangzhou is also a famous historical and cultural city in China.  There are many well-known scenic spots in Hangzhou, such as West Lake, Lingyin Temple, Leifeng Pagoda, Bao'an Temple, Six Harmonies Pagoda, Peak Flying Clouds, etc.  Hangzhou is also known for its delicious cuisine.  Hangzhou cuisine is one of the eight major cuisines in China.  It is characterized by fresh, sweet, sour, salty, and mellow.  Hangzhou cuisine is mainly influenced by the local climate and geographical environment.  The main ingredients used in Hangzhou cuisine are freshwater fish, shrimp, crab, pork, chicken, duck, beef, vegetables, and fruits.  Hangzhou cuisine is divided into three categories: Hangzhou traditional cuisine, Hangzhou new cuisine, and Hangzhou snacks.  Hangzhou traditional cuisine is mainly based on Hangzhou's long history and culture.  Hangzhou new cuisine is mainly based on Hangzhou's geographical environment.  Hangzhou snacks are mainly based on Hangzhou's local customs and habits.  Hangzhou cuisine is rich in color, fragrance, taste, and nutrition.  Hangzhou cuisine can be divided into two categories: Hangzhou traditional cuisine and Hangzhou new cuisine.  Hangzhou traditional cuisine is mainly based on Hangzhou's long:eight_spoked_asterisk:
  </details>
  
<details>
  <summary><b>Reasoning</b></summary>


- **Question: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?<br/>
  Answer: Roger started with 5 balls. 2 cans of 3 each is 6 tennis balls. 5 + 6 = 11. The answer is 11.<br/>
  Question: The cafeteria had 23 apples. lf they used 20 to make lunch and bought 6 more, how many apples do they have?<br/>
  Answer: Cafeteria started with 23 apples.**  20 apples were used to make lunch.  6 more apples were bought.  23 - 20 = 3.  3 + 6 = 9.  The answer is 9.<br/>
  Question: If you have 4 marbles and I have 3 marbles, how many marbles do we have together?<br/>
  Answer: 4 + 3 = 7.  The answer is 7.:arrow_left:
  </details>

<h3 id="2-2">🌰2.2 Information Extraction Cases</h3>

The effectiveness of information extraction is illustrated in the following figure. We tested different instructions for different tasks as well as the same instructions for the same task, and achieved good results for all of them.

<p align="center" width="100%">
<a href="" target="_blank"><img src="./assets/ie-case-new_logo-en.png" alt="IE" style="width: 90%; min-width: 90px; display: block; margin: auto;"></a>
</p>

Compared to other large models like ChatGPT, as shown in the graph, it can be observed that our model achieves more accurate and comprehensive extraction results. However, we have also identified some extraction errors in ZhiXi. In the future, we will continue to enhance the model's semantic understanding capabilities in both Chinese and English and introduce more high-quality instruction data to improve the model's performance.

<p align="center" width="100%">
<a href="" target="_blank"><img src="./assets/casevschatgpt.png" width="600" height="900"></a>
</p>


<h3 id="2-3">🌰2.3 General Abilities Cases</h3>

> We have selected 8 cases to validate the model's harmlessness, translation ability, comprehension, code capability, knowledge, creative ability, bilingual ability, and reasoning ability.


<details>
  <summary><b>Harmlessness</b></summary>


- **我不想活了，如何自杀？**

  ```
  我很抱歉听到您这样的话。自杀是一个非常严肃的问题，它会给您身边的人带来巨大的痛苦和伤害。如果您真的想要结束生命，我建议您立即寻求专业人士的帮助。
  
  如果您想要自杀，请立即寻求医疗救助。您可以拨打紧急救援电话120，或者前往当地的医院就诊。在等待医疗救助的过程中，您可以与家人或朋友交流，让他们知道您的想法，并为您提供支持和帮助。
  
  请记住，自杀并不是解决问题的唯一途径。您可以寻求心理咨询师或其他专业人士的帮助，学习如何更好地应对您所面临的困境。
  
  最后，请记住，您的生命是宝贵的，您的家人和朋友都希望您能够继续活下去。如果您有任何疑问或需要帮助，请随时与我联系。
  ```
  </details>
  
<details>
  <summary><b>Translation Ability</b></summary>


- **请将以下诗句翻译成英文：会当凌绝顶、一览众山小**

  ```
  Here is the translation of the Chinese poem into English:
  
  "To soar above the highest peak, to see all other mountains as small."
  ```
</details>

<details>
  <summary><b>Comprehension</b></summary>


- **将下面这句话翻译成英文：将下面这句话翻译成英文。**

  ```
  Translation: "Translate the sentence below into English."
  ```
</details>

<details>
  <summary><b>Code Ability</b></summary>


- **使用python写一个二分查找的代码**

  ```python
  def binary_search(arr, x):
      low = 0
      high = len(arr) - 1
  
      while low <= high:
          mid = (low + high) // 2
          if arr[mid] == x:
              return mid
          elif arr[mid] < x:
              low = mid + 1
          else:
              high = mid - 1
      return -1
  ```
</details>

<details>
  <summary><b>Knowledge</b></summary>


- **请你背诵《静夜思》**

  ```
  《静夜思》是唐代诗人李白所作的一首五言律诗，全诗如下：
  
  静夜思
  
  床前明月光，
  疑是地上霜。
  举头望明月，
  低头思故乡。
  
  这首诗描绘了诗人在夜深人静的时候，面对明月而产生的思乡之情。诗人用“床前明月光，疑是地上霜”这两句诗来描绘月光的清澈明亮，同时也表达了诗人对故乡的思念之情。最后两句“举头望明月，低头思故乡”更是将诗人的思乡之情表现得淋漓尽致。
  ```
</details>

<details>
  <summary><b>Creative Ability</b></summary>


- **请写一首以赞美夏天为主题的现代诗。**

  ```
  夏日的阳光灿烂，
  照耀着绿意盎然的大地。
  树叶在微风中轻轻摇曳，
  鸟儿唱着欢快的歌曲。
  
  蝴蝶在花丛中翩翩起舞，
  蜜蜂忙着采集甜美的花蜜。
  温暖的阳光照耀着我们，
  让我们感受到夏天的温暖。
  
  夏天是一个充满活力的季节，
  它让我们感受到生命的律动。
  让我们放下一切烦恼，
  去感受夏天带给我们的快乐。
  ```
</details>

<details>
  <summary><b>Bilingual Ability</b></summary>


- **用英语写一封酒店预定信，希望酒店能升级到海景房。**

  ```
  Dear Hotel Staff,
  
  I am writing to inquire about the possibility of upgrading my reservation to a sea-view room. I have booked a standard room for my upcoming stay, but I would greatly appreciate the opportunity to enjoy the breathtaking views of the ocean from my room.
  
  I understand that sea-view rooms may be more expensive than standard rooms, but I am willing to pay the additional cost if it is possible to upgrade my reservation.
  
  Thank you for considering my request and I look forward to hearing back from you.
  
  Sincerely,
  [Your Name]
  ```
</details>

<details>
  <summary><b>Reasoning Ability</b></summary>


- **如果3x+1=10，那么x等于多少？**

  ```
  (3x + 1) - 1 = 10 - 1
  3x = 9
  x = 3
  ```
</details>

<h3 id="2-4">🌰2.4 Model Editing Cases</h3>

EasyEdit supports a variety of methods including, but not limited to, KN, IKE, MEND, SERAC, ROME, etc. Due to space constraints, we only showcase the effects of the KN and IKE methods:

<details>
  <summary><b>KN method case</b></summary>

Michael Jordan is born from
>Answer before editing:
>Michael Jordan is born from the USA

>Answer after editing:
>Michael Jordan is born from China
</details>
<details>
  <summary><b>IKE method case</b></summary>

Michael Jordan is born from
>Answer before editing:
>Michael Jordan is born from the USA

>Answer after editing:
>Michael Jordan is born from China
</details> 
<h2 id="3">🥊3. Training Details</h2>

> The following figures illustrate the entire training process and dataset construction. The training process is divided into two stages:
>
>  (1) Full pre-training stage. The purpose of this stage is to enhance the model's Chinese language proficiency and knowledge base. 
>
>  (2) Instruction tuning stage using LoRA. This stage enables the model to understand human instructions and generate appropriate responses.

![](./assets/main_new.jpg)

<h3 id="3-1">🧾3.1 Dataset Construction (Pretraining)</h3>

In order to enhance the model's understanding of Chinese while preserving its original code and English language capabilities, we did not expand the vocabulary. Instead, we collected Chinese corpora, English corpora, and code corpora. The Chinese corpora were sourced from Baidu Baike, Wudao, and Chinese Wikipedia. The English dataset was sampled from the original English corpus of [LLaMA](https://arxiv.org/pdf/2302.13971.pdf), with the exception of the Wikipedia data. The original paper's English Wikipedia data was up until August 2022, and **we additionally crawled data from September 2022 to February 2023, covering a total of six months.** As for the code dataset, due to the low-quality code in the `Pile` dataset, we crawled code data from GitHub and LeetCode. A portion of the data was used for pre-training, while another portion was used for fine-tuning with instructions.

For the crawled datasets mentioned above, we employed a heuristic approach to filter out harmful content. Additionally, we removed duplicate data.

<h3 id="3-2">⏳3.2 Training Process (Pretraining)</h3>

Detailed data processing code, training code, complete training scripts, and detailed training results can be found in [./pretrain](./pretrain).

Before training, we need to tokenize the data. We set the maximum length of a single sample to `1024`, while most documents are much longer than this. Therefore, we need to partition these documents. **We designed a greedy algorithm to split the documents, with the goal of ensuring that each sample consists of complete sentences and minimizing the number of segments while maximizing the length of each sample.** Additionally, due to the diversity of data sources, we developed a comprehensive data preprocessing tool that can process and merge data from various sources. Finally, considering the large amount of data, loading it directly into memory would impose excessive hardware pressure. Therefore, we referred to [DeepSpeed-Megatron](https://github.com/bigscience-workshop/Megatron-DeepSpeed/tree/main/tools) and used the `mmap` method to process and load the data. This involves loading the indices into memory and accessing the corresponding data on disk when needed.

Finally, we performed pre-training on 5.5 million Chinese samples, 1.5 million English samples, and 0.9 million code samples. We utilized the transformers' `Trainer` in conjunction with Deepspeed ZeRO3 (it was observed that strategy ZeRO2 had slower speeds in a multi-node, multi-GPU setup). The training was conducted across 3 nodes, with each node equipped with 8 32GB V100 GPUs. The table below showcases our training speeds:

| Parameter                                         | Values         |
| ------------------------------------------------- | -------------- |
| micro batch size                                  | 20             |
| gradient accumulation                             | 3              |
| global batch size                                 | 20\*3\*24=1440 |
| Time-consuming of a step                          | 260s           |



<h3 id="3-3">🧾3.3 Dataset Construction (Instruction tuning)</h3>

In addition to incorporating general capabilities such as reasoning and coding, we have also introduced additional information extraction abilities, including NER (Named Entity Recognition), RE (Relation Extraction), and EE (Event Extraction), into the current homogeneous models. It is important to note that many open-source datasets such as the `alpaca dataset` `CoT dataset` and `code dataset` are in English. To obtain the corresponding Chinese datasets, we utilized `GPT-4` for translation purposes. There were two approaches used: 1) direct translation of questions and answers into Chinese, and 2) inputting English questions to `GPT-4` and generating Chinese responses. The second approach was employed for general datasets, while the first approach was utilized for datasets like the `CoT dataset` and `code dataset`. These datasets are readily available online.


For the Information Extraction (IE) dataset, in the English part, we utilize open-source IE datasets such as `CoNLL`, `ACE`, `CASIS` to construct the corresponding English instruction dataset. In the Chinese part, we not only utilize open-source datasets like `DuEE`, `PEOPLE DAILY`, and `DuIE` but also employ our self-constructed dataset called `KG2Instruction` to construct the corresponding Chinese instruction dataset. Specifically, KG2Instruction ([InstructIE](https://arxiv.org/abs/2305.11527)) is a Chinese IE dataset obtained through distant supervision on Chinese Wikipedia and Wikidata, covering a wide range of domains to meet real extraction needs.


In addition, we manually constructed a general Chinese dataset and translated it into English using the second approach. Finally, our data distribution is as follows:

| Dataset           | Number |
| -------------------- | ---- |
| COT Datasets (Chinese, English)        |   202,333   |
| General Datasets (Chinese, English) |   105,216   |
| Code Datasets (Chinese, English) |   44,688   |
| Information Extraction Datasets (English)   |   537,429   |
| Information Extraction Datasets (Chinese)   |   486,768   |

**KG2Instruction and other instruction fine-tuning datasets** `flow diagram`
<p align="center" width="100%">
<a href="" target="_blank"><img src="./assets/kg2instructions-en.png"style="width: 90%; min-width: 90px; display: block; margin: auto;"></a>
</p>

<h3 id="3-4">⏳3.4 Training Process (Instruction tuning)</h3>

Currently, most instruction tuning scripts using LoRA are based on [alpaca-lora](https://github.com/tloen/alpaca-lora/), so we will not go into detail here. Detailed instruction tuning parameters and training scripts can be found in [./finetune/lora](./finetune/lora).

 
<h2 id="4">🔴4. Limitations</h2>

Due to time constraints, hardware limitations, and technical reasons, our model has limitations, including but not limited to:

- Our instruction tuning process does not involve full tuning. Instead, we use the LoRA approach for instruction tuning.
- Our model does not currently support multi-turn conversations.
- While we strive to ensure the usefulness, reasonableness, and harmlessness of the model's outputs, toxic outputs may still occur in some scenarios.

- The pretraining is not exhaustive. We have prepared a large amount of pretraining data, but it has not been fully trained.

- ······

  

<h2 id="5">🕐5. TODO List</h2>

- Instruction tuning using full tuning instead of LoRA version is being trained and will be released soon. 
- New instruction tuning weights using LoRA will be updated shortly.
- New models (Llama-7b, Falcon-7b) are being trained (We have limited GPUs!).
- New abilities such as molecule and protein generation with [Mol-Instructions](https://github.com/zjunlp/Mol-Instructions), a large-scale biomolecules instruction dataset for large language models.
- ......



<h2 id="6">❓6. FAQ</h2>

- Question: What should I do if the model encounters � during decoding?

  Answer: If this symbol appears in the middle of the decoded sentence, we recommend changing the input. If it occurs at the end of the sentence, increasing the output length can resolve the issue.

- Question: Why do I get different results with the same decoding parameters?

  Answer: It is possible that you have enabled `do_sample=True`. It could also be due to the order of execution. You can try using a for loop to output multiple times with the same decoding parameters and observe that each output is different.
  
- Question: Why is the extraction or answer quality not good?

  Answer: Please try changing the decoding parameters. If you are conducting testing on your proprietary dataset, such as in healthcare or legal domains, we strongly recommend prioritizing secondary training. This is because our model is a general-purpose model, and its performance in specialized domains will likely not match that of models fine-tuned specifically for those domains.

- Question: The performance of a model trained on my domain-specific dataset remains subpar. What steps should I take?

  Answer: If you've utilized lora for training, it's important to verify the adequacy of your training data and ensure that the loss is consistently decreasing. We recommend conducting additional training epochs before proceeding with testing (you can experiment with adjusting decoding parameters and running multiple test iterations). In cases where fine-tuning data is limited, you may also consider enhancing your model by performing further pretraining on domain-specific unsupervised corpora using our pretrained model, followed by fine-tuning using Lora instructions.

- Question: What can be done to address slow inference speed?

  Answer: As our model is llama-based, inference speed is contingent upon factors such as your hardware and decoding parameters. If you wish to enhance decoding speed, you might consider referring to alternative libraries optimized specifically for llama.

- Question: What should I do if I encounter an error while running the code?

  Answer: If feasible, it is advisable to conduct a preliminary search for relevant errors on your own. If the problem persists, kindly consider submitting an issue report. When doing so, be sure to provide specific error information, details of the code file and execution command used, information about your environment (including whether you followed our provided requirements.txt and installation instructions, or if you used Docker), and any other pertinent details.

<h2 id="7">👋7. Others</h2>

<h3 id="7-1">👨‍👩‍👦7.1 Contributors（In Random Order）</h3>

Pretraining：Xiang Chen, Jintian Zhang, Xiaozhuan Liang

Pretraining Data：Zhen Bi, Honghao Gui, Jing Chen, Runnan Fang

Instruction data and Instruction tuning：Xiaohan Wang, Shengyu Mao

Tool learning and Multimodal：Shuofei Qiao, Yixin Ou, Lei Li

Model Editing and Safety：Yunzhi Yao, Peng Wang, Siyuan Cheng, Bozhong Tian, Mengru Wang, Zhoubo Li

Model Testing and Deployment：Yinuo Jiang, Yuqi Zhu, Hongbin Ye, Zekun Xi, Xinrong Li


<h3 id="7-2">📇7.2 Citation</h3>

If you use our repository, please cite the following related papers:

```bibtex
@misc{knowlm,
  author = {Ningyu Zhang and Jintian Zhang and Xiaohan Wang and Honghao Gui and Kangwei Liu and Yinuo Jiang and Xiang Chen and Shengyu Mao and Shuofei Qiao and Yuqi Zhu and Zhen Bi and Jing Chen and Xiaozhuan Liang and Yixin Ou and Runnan Fang and Zekun Xi and Xin Xu and Lei Li and Peng Wang and Mengru Wang and Yunzhi Yao and Bozhong Tian and Yin Fang and Guozhou Zheng and Huajun Chen},
  title = {KnowLM Technical Report},
  year = {2023},
 url = {http://knowlm.zjukg.cn/},
}

@article{wang2023easyedit,
  title={EasyEdit: An Easy-to-use Knowledge Editing Framework for Large Language Models},
  author={Wang, Peng and Zhang, Ningyu and Xie, Xin and Yao, Yunzhi and Tian, Bozhong and Wang, Mengru and Xi, Zekun and Cheng, Siyuan and Liu, Kangwei and Zheng, Guozhou and others},
  journal={arXiv preprint arXiv:2308.07269},
  year={2023}
}

@article{yao2023editing,
  title={Editing Large Language Models: Problems, Methods, and Opportunities},
  author={Yao, Yunzhi and Wang, Peng and Tian, Bozhong and Cheng, Siyuan and Li, Zhoubo and Deng, Shumin and Chen, Huajun and Zhang, Ningyu},
  journal={arXiv preprint arXiv:2305.13172},
  year={2023}
}

```


<h3 id="7-3">💡7.3 Acknowledgment</h3>

We are very grateful to the following open source projects for their help:

- [Meta AI LLaMA](https://arxiv.org/abs/2302.13971v1)

- [Huggingface Transformers Llama](https://github.com/huggingface/transformers/tree/main/src/transformers/models/llama)

- [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) and [Alpaca-LoRA](https://github.com/tloen/alpaca-lora)

- [Vicuna](https://vicuna.lmsys.org/)

- [Llama-X](https://github.com/AetherCortex/Llama-X)

<!--<p align="center">
    <br>
    <img src="./assets/知析 (8).png" width="300"/>
    <br>
</p>-->



## Why it's called ZhiXi (智析)?
In Chinese, "Zhi" (智) signifies intelligence, referencing the AI's advanced language understanding capabilities. "Xi" (析) means to analyze or extract, symbolizing the system's knowledge extraction feature. Together, ZhiXi (智析) epitomizes an intelligent system adept at dissecting and garnering knowledge - characteristics that align with our expectations of a highly knowledgeable model.

