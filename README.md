<div align="center">
 <img src="imgs/logo.png" alt="RWKV Survey" width="500"><br>
  A collection of papers and resources related to a survey of RWKV.
</div>
<br>
 
<p align="center">
  Zhiyuan Li<sup>*1</sup>&nbsp&nbsp
  Tingyu Xia<sup>*1</sup>&nbsp&nbsp
  Yi Chang<sup>1</sup>&nbsp&nbsp
  Yuan Wu<sup>#1</sup>&nbsp&nbsp
</p>  
<p align="center">
<sup>1</sup> Jilin University<br>
(*: Co-first authors, #: Corresponding author)
</p>

# Papers and resources for RWKV

The papers are organized according to our survey: [A Survey of RWKV](https://arxiv.org/abs/2412.14847).

**NOTE:** As we cannot update the arXiv paper in real time, please refer to this repo for the latest updates and the paper may be updated later. We also welcome any pull request or issues to help us make this survey perfect. Your contributions will be acknowledged in <a href="#acknowledgements">acknowledgements</a>.

Related projects:

- RWKV-IR: [[Exploring Real&Synthetic Dataset and Linear Attention in Image Restoration](https://arxiv.org/abs/2412.03814)]

![](imgs/framework_new.png)

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#news-and-updates">News and Updates</a></li>
    <li><a href="#rwkv">RWKV</a>
      <ul>
        <li><a href="#rwkv-papers-and-official-repository">RWKV papers and official repository</a></li>
        <li><a href="#various-implementations">Various Implementations</a></li>
      </ul>  
    </li>
    <li>
      <a href="#applications-of-the-rwkv-model">Applications of the RWKV model</a>
      <ul>
        <li><a href="#natural-language-generation">Natural Language Generation</a></li>
        <li><a href="#natural-language-understanding">Natural Language Understanding</a></li>
        <li><a href="#other-nlp-tasks">Other NLP tasks</a></li>
        <li><a href="#computer-vision">Computer Vision</a></li>
        <li><a href="#audio-and-music-tasks">Audio and Music Tasks</a></li>
        <li><a href="#web-application">Web Application</a></li>
        <li><a href="#evaluation-of-rwkv-models">Evaluation of RWKV Models</a></li>
        <li><a href="#others">Others</a></li>
      </ul>
    </li>
    <li><a href="#Contributing">Contributing</a></li>
    <li><a href="#citation">Citation</a></li>
    <li><a href="#acknowledgements">Acknowledgments</a></li>
  </ol>
</details>

## News and updates

- [19/12/2024] The first version of the paper is released on arXiv: [A Survey of RWKV](https://arxiv.org/abs/2412.14847).

## RWKV

### RWKV papers and official repository

1. RWKV: Reinventing RNNs for the Transformer Era 2023.[[paper](https://arxiv.org/abs/2305.13048)]
2. Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence 2024. [[paper](https://arxiv.org/abs/2404.05892)]
3. RWKV official repository [[project](https://github.com/BlinkDL/RWKV-LM)]

### Various Implementations

1. INT4/INT5/INT8 and FP16 inference on CPU for RWKV language model [[project](https://github.com/RWKV/rwkv.cpp)]
2. Inference a RWKV model in pure C [[project](https://github.com/kong13661/rwkv.c)]
3. A torchless, c++ rwkv implementation using 8bit quantization [[project](https://github.com/harrisonvanderbyl/rwkv-cpp-accelerated)]
4. RWKV model running on windows with C++ [[project](https://github.com/ZeldaHuang/rwkv-cpp-server)]
5. RWKV tokenizer based on cpp [[project](https://github.com/szxysdt/rwkv_tokenizer_cpp)]
6. Implementation of the RWKV language model in pure WebGPU/Rust [[project](https://github.com/cryscan/web-rwkv)]
7. Rust implements RWKV [[project](https://github.com/mrsteyk/rwkvk-rs)]
8. A fast RWKV Tokenizer written in Rust [[project](https://github.com/cahya-wirawan/rwkv-tokenizer)]
9. Implementation of RWKV in Rust [[project](https://github.com/KerfuffleV2/smolrsrwkv)]
10. An rust wrapper of rwkv-cpp [[project](https://github.com/yorkzero831/rwkv-rust)]
11. RWKV ported to tensorflow [[project](https://github.com/RWKV/rwkv-tensorflow)]
12. RWKV Tensorflow2.0 [[project](https://github.com/123mbcz123/RWKV5-Tensorflow2.0) [project](https://github.com/123mbcz123/RWKV-Tensorflow2.0) [project](https://github.com/breadbread1984/RWKV-tf2)]
13. A TensorFlow implementation of the RWKV layer [[project](https://github.com/andvid72/Tensor-RWKV) [project](https://github.com/prouast/tensorflow-rwkv)]
14. Tensorflow bindings for RWKV [[project](https://github.com/rj42/rwkv-cuda)]
15. Port of the RWKV-LM model in Fortran [[project](https://github.com/nlpodyssey/rwkv.f90)]
16. Implementation of RWKV in Julia [[project](https://github.com/guanyilun/rwkv.jl)]
17. Minimal RWKV implementation in Zig [[project](https://github.com/metalwhale/rwkv.zig)]
18. Implementation of RWKV in Go [[project](https://github.com/nlpodyssey/rwkv) [project](https://github.com/ZaraK0/rwkvalgorithm)]
19. Pure go for rwkv and support cross-platform [[project](https://github.com/seasonjs/rwkv)]
20. A go wrapper around the rwkv.cpp library [[project](https://github.com/donomii/go-rwkv.cpp)]
21. VerbaFlow, a neural architecture written in Go designed specifically for language modeling tasks [[project](https://github.com/nlpodyssey/verbaflow)]
22. RWKV World tokenizer for Go [[project](https://github.com/Ronsor/rwkv-tokenizer-go)]
23. Implementation of RWKV in Keras [[project](https://github.com/pass-lin/RWKV6-Keras) [project](https://github.com/jackd/keras-rwkv)]
24. The RWKV Language Model Inference on PaddlePaddle [[project](https://github.com/HighCWu/rwkv-paddle)]
25. Paddle version RWKV-v2-RNN [[project](https://github.com/JunnYu/RWKV-v2-RNN-paddle)]
26. TPU RWKV Implementations [[project](https://github.com/tensorpro/tpu_rwkv)]
27. RWKV-LM in JAX and Haiku [[project](https://github.com/mrsteyk/RWKV-LM-jax)]
28. A JAX implementation of the RWKV [[project](https://github.com/saran-gangster/rwkv-jax)]
29. Node.js implementation binding for the RWKV.cpp module [[project](https://github.com/RWKV/RWKV-cpp-node)]
30. Native Node.js tokenizer for RWKV [[project](https://github.com/RWKV/RWKV-tokenizer-node)]
31. Implementation of RWKV using mlx [[project](https://github.com/dc-dc-dc/mlx-rwkv)]
32. RWKV v5, v6 LoRA Trainer on Cuda and Rocm Platform [[project](https://github.com/OpenMOSE/RWKV5-LM-LoRA)]
33. RWKV v5, v6 infctx LoRA trainer [[project](https://github.com/OpenMOSE/RWKV-infctx-trainer-LoRA)]
34. RWKV Infinite Context trainer [[project](https://github.com/RWKV/RWKV-infctx-trainer)]
35. A lightweight RWKV inference platform [[project](https://github.com/OpenMOSE/RWKV-Infer)]
36. Run RWKV V4 ONNX using the Android cpu [[project](https://github.com/ZTMIDGO/RWKV-Android)]
37. Inference RWKV on NCNN [[project](https://github.com/MollySophia/rwkv-ncnn)]
38. Inference RWKV v5, v6 and (WIP) v7 with Qualcomm AI Engine Direct SDK [[project](https://github.com/MollySophia/rwkv-qualcomm)]
39. JNI wrapper for rwkv.cpp [[project](https://github.com/vaccovecrana/rwkv.jni)]
40. An RWKV6 operator designed for Keras3 [[project](https://github.com/infiy-quine/RWKV6_Keras_Operator)]
41. An API based on Flask implementation of RWKV_Role_Playing project [[project](https://github.com/shengxia/RWKV_Role_Playing_API)]
42. The nanoGPT-style implementation of RWKV Language Model [[project](https://github.com/Hannibal046/nanoRWKV)]

## Applications of the RWKV model

### Natural Language Generation

1. Combining information retrieval and large language models for a chatbot that generates reliable, natural-style answers. [[paper](https://ceur-ws.org/Vol-3630/LWDA2023-paper27.pdf)]
2. AI-Writer [[project](https://github.com/BlinkDL/AI-Writer)]
3. RWKV chatbot [[project](https://github.com/harrisonvanderbyl/rwkv_chatbot)]
4. RWKV wechat bot [[project](https://github.com/averyyan/RWKV-wechat-bot)]
5. RWKV chat command line [[project](https://github.com/az13js/rwkv_chat_command_line)]
6. A QQ Chatbot based on RWKV [[project](https://github.com/cryscan/eloise)]
7. Local lightweight chat AI based on RWKV [[project](https://github.com/bilibini/Meow-AI)]
8. Espritchatbot RASA RWKV [[project](https://github.com/kimou6055/Espritchatbot-RASA-RWKV)]
9. Espritchatbot RASA RAVEN [[project](https://github.com/karim-aloulou/Espitchatbot-RASA-RAVEN)]
10. RAG system for RWKV [[project](https://github.com/AIIRWKV/RWKV-RAG)]
11. ChatRWKV in wechat [[project](https://github.com/MrTom34/ChatRWKV-in-wechat-Version-1)]
12. Generating WeChat replies using the RWKV runner [[project](https://github.com/LeoLin4258/Infofusion)]
13. RWKV-Drama [[project](https://github.com/ms-KuroNeko/RWKV-Drama)]
14. RWKV Role Playing with GPT SoVITS [[project](https://github.com/v3ucn/RWKV_Role_Playing_with_GPT-SoVITS)]
15. A Telegram LLM bot [[project](https://github.com/spion/notgpt)]
16. Chatbots based on nonebot and RWKV [[project](https://github.com/123summertime/ykkz)]
17. Online chat rooms based on PyWebIO and RWKV models [[project](https://github.com/No-22-Github/Easy_RWKV_webui)]

### Natural Language Understanding

1. An approach to mongolian neural machine translation based on rwkv language model and contrastive learning [[paper](https://link.springer.com/chapter/10.1007/978-981-99-8132-8_25)]
2. Onlysportslm: Optimizing sports-domain language models with sota performance under billion parameters [[paper](https://arxiv.org/abs/2409.00286)]
3. Virtual Assistant [[project](https://github.com/samttoo22-MewCat/lala_rwkv_chatbot_2.0)]
4. PDF Query Systems [[project](https://github.com/ck-unifr/pdf_parsing)]
5. A classification model using RWKV [[project](https://github.com/yynil/RWKV-Classification)]
6. Novel continuation model based on RWKV [[project](https://github.com/jiawanfan-yyds/novel-rwkv_demo)]
7. A large ai town built on RWKV [[project](https://github.com/recursal/ai-town-rwkv-proxy)]
8. Questions and Answers based on RWKV [[project](https://github.com/seitzquest/RavenWhisperer)]
9. RWKV using wenda to QA and ICL [[project](https://github.com/xiaol/wenda-RWKV)]
10. A comprehensive mobile application based on RWKV [[project](https://github.com/khhaliil/AVATARIO)]
11. Knowledge graph extraction tool based on RWKV [[project](https://github.com/Ojiyumm/rwkv_kg)]

### Other NLP Tasks

1. Multi-scale rwkv with 2-dimensional temporal convolutional network for short-term photovoltaic power forecasting [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0360544224028433)]
2. Contrastive learning for clinical outcome prediction with partial data sources [[paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11326519/)]
3. Stock prediction using RWKV [[project](https://github.com/tomer9080/Stock-Prediction-Using-RWKV)]
4. Dft: A dual-branch framework of fluctuation and trend for stock price prediction [[paper](https://arxiv.org/abs/2411.06065)]
5. Matcc: A novel approach for robust stock price prediction incorporating market trends and cross-time correlations [[paper](https://dl.acm.org/doi/abs/10.1145/3627673.3679715)]
6. A code completion model based rwkv with bimodal pretraining [[paper](https://www.researchsquare.com/article/rs-3387525/v1)]
7. Rwkv-based encoder-decoder model for code completion [[paper](https://ieeexplore.ieee.org/abstract/document/10442108/)]
8. Experimentation in content moderation using rwkv [[paper](https://arxiv.org/abs/2409.03939)]
9. Goldfinch: High performance rwkv/transformer hybrid with linear pre-fill and extreme kv-cache compression [[paper](https://arxiv.org/abs/2407.12077)]
10. Rwkv-ts: Beyond traditional recurrent neural network for time series tasks [[paper](https://arxiv.org/abs/2401.09093)]
11. Temporal and interactive modeling for efficient human-human motion generation [[paper](https://arxiv.org/abs/2408.17135)]
12. Rrwkv: capturing long-range dependencies in rwkv [[paper](https://arxiv.org/abs/2306.05176)]
13. Lkpnr: Large language models and knowledge graph for personalized news recommendation framework [[paper](https://search.ebscohost.com/login.aspx?direct=true&profile=ehost&scope=site&authtype=crawler&jrnl=15462218&AN=178256380&h=mPC2JIgqSZw4phTzIrP%2FKqjs9uCWP6JzGqQAI5ecEQmASbdVuYmY%2BQ17K27Xqqb%2BBbDDdbl%2F6scZRZNvhqBfCg%3D%3D&crl=c)]
14. Optimizing robotic manipulation with decision-rwkv: A recurrent sequence modeling approach for lifelong learning [[paper](https://arxiv.org/abs/2408.01950)]
15. Prosg: Using prompt synthetic gradients to alleviate prompt forgetting of rnn-like language models [[paper](https://arxiv.org/abs/2311.01981)]
16. Spikegpt: Generative pre-trained language model with spiking neural networks [[paper](https://arxiv.org/abs/2302.13939)]
17. General population projection model with census population data [[paper](https://scholarworks.lib.csusb.edu/etd/1803/)]
18. Enhancing transformer rnns with multiple temporal perspectives [[paper](https://arxiv.org/abs/2402.02625)]
19. Sensorimotor attention and language-based regressions in shared latent variables for integrating robot motion learning and llm [[paper](https://arxiv.org/abs/2407.09044)]
20. A transfer learning-based training approach for dga classification [[paper](https://link.springer.com/chapter/10.1007/978-3-031-64171-8_20)]

### Computer Vision

1. Bsbp-rwkv: Background suppression with boundary preservation for efficient medical image segmentation [[paper](https://dl.acm.org/doi/abs/10.1145/3664647.3681033)]
2. Restore-rwkv: Efficient and effective medical image restoration with rwkv [[paper](https://arxiv.org/abs/2407.11087)]
3. Rnn-based multiple instance learning for the classification of histopathology whole slide images [[paper](https://link.springer.com/chapter/10.1007/978-981-97-1335-6_29)]
4. Lion: Linear group rnn for 3d object detection in point clouds [[paper](https://arxiv.org/abs/2407.18232)]
5. Pointrwkv: Efficient rwkv-like model for hierarchical point cloud learning [[paper](https://arxiv.org/abs/2405.15214)]
6. Occrwkv: Rethinking efficient 3d semantic occupancy prediction with linear complexity [[paper](https://arxiv.org/abs/2409.19987)]
7. Tls-rwkv: Real-time online action detection with temporal label smoothing [[paper](https://link.springer.com/article/10.1007/s11063-024-11540-0)]
8. From explicit rules to implicit reasoning in an interpretable violence monitoring system [[paper](https://arxiv.org/abs/2410.21991)]
9. Hybrid recurrent-attentive neural network for onboard predictive hyperspectral image compression [[paper](https://ieeexplore.ieee.org/abstract/document/10641584/)]
10. Mamba or rwkv: Exploring high-quality and high-efficiency segment anything model [[paper](https://arxiv.org/abs/2406.19369)]
11. Vision-rwkv: Efficient and scalable visual perception with rwkv-like architectures [[paper](https://arxiv.org/abs/2403.02308)]
12. Visualrwkv-hd and uhd: Advancing high-resolution processing for visual language models [[paper](https://arxiv.org/abs/2410.11665)]
13. Video rwkv: Video action recognition based rwkv [[paper](https://arxiv.org/abs/2411.05636)]
14. Rwkv-clip: A robust vision-language representation learner [[paper](https://arxiv.org/abs/2406.06973)]
15. Sdit: Spiking diffusion model with transformer [[paper](https://arxiv.org/abs/2402.11588)]
16. Social-cvae: Pedestrian trajectory prediction using conditional variational auto-encoder [[paper](https://link.springer.com/chapter/10.1007/978-981-99-8132-8_36)]
17. Diffusion-rwkv: Scaling rwkv-like architectures for diffusion models [[paper](https://arxiv.org/abs/2404.04478)]
18. Exploring real&synthetic dataset and linear attention in image restoration [[paper](https://arxiv.org/abs/2412.03814)]
19. Facial Expression Recognition with RWKV Architecture [[project](https://github.com/lukasVierling/FaceRWKV)]
20. Image denoising model based on rwkv [[project](https://github.com/lll143653/rwkv-denoise)]

### Audio and Music Tasks

1. Android RWKV MIDI [[project](https://github.com/ZTMIDGO/Android-RWKV-MIDI)]
2. Use RWKV to generate symbolic music to a text file. [[project](https://github.com/patchbanks/RWKV-v4-MIDI)]
3. Use the RWKV-4 music model to generate the texture and music [[project](https://github.com/agreene5/Procedural-Purgatory)]
4. Exploring rwkv for memory efficient and low latency streaming asr [[paper](https://arxiv.org/abs/2309.14758)]
5. Speech missions with frozen RWKV language models [[project](https://github.com/AGENDD/RWKV-ASR)]
6. AudioRWKV: Pretrained Audio RWKV for Audio Pattern Recognition [[project](https://github.com/diggerdu/AudioRWKV)]
7. Why perturbing symbolic music is necessary: Fitting the distribution of never-used notes through a joint probabilistic diffusion model [[paper](https://arxiv.org/abs/2408.01950)]
8. RWKV Based Music Generator [[project](https://github.com/asuller/RWKV-MusicGenerator)]
9. Music Genre Classification RWKV [[project](https://github.com/AverageJoe9/Music-Genre-Classification-RWKV)]
10. Advancing vad systems based on multi-task learning with improved model structures [[paper](https://arxiv.org/abs/2312.14860)]

### Web Application

1. Web api based on rwkv.cpp [[project](https://github.com/YuChuXi/MoZi-RWKV)]
2. RWKV Webui GPT-SoVITS [[project](https://github.com/v3ucn/RWKV_3B_7B_Webui_GPT-SoVITS)]
3. AI00 RWKV Server [[project](https://github.com/Ai00-X/ai00_server)]
4. RWKV-4 running in the browser [[project](https://github.com/josephrocca/rwkv-v4-web)]
5. Role-playing webui based on RWKV [[project](https://github.com/shengxia/RWKV_Role_Playing)]
6. RWKV QQBot BackEnd [[project](https://github.com/yuunnn-w/RWKV_QQBot_BackEnd)]
7. A axum web backend for web-rwkv [[project](https://github.com/Prunoideae/web-rwkv-axum)]
8. ChatGPT-like Web UI for RWKVstic [[project](https://github.com/hizkifw/WebChatRWKVstic)]
9. Use chatux to make chatRWKV a web chatbot [[project](https://github.com/riversun/chatux-server-rwkv)]
10. Flask frame based chatbot server [[project](https://github.com/t4wefan/ChatRWKV-flask-api)]
11. ChatRWKV webui [[project](https://github.com/StarDreamAndFeng/ChatRWKV-webui)]
12. Flask server for RWKV [[project](https://github.com/RafaRed/RWKV-api)]
13. rwkv.cpp webui Macos [[project](https://github.com/v3ucn/rwkv.cpp_webui_Macos)]
14. rwkv.cpp webui GPT-SoVITS [[project](https://github.com/v3ucn/rwkv.cpp_webui_GPT-SoVITS)]
15. RWKV Role Playing Web UI [[project](https://github.com/shengxia/RWKV_Role_Playing_UI)]

### Evaluation of RWKV Models

In this paper, we have assembled 17 benchmark tests, each highlighting distinct facets and criteria for evaluation.

**NOTE:** We may miss some evaluations. Your suggestions are highly welcomed!

| Benchmark                                                                   | Focus                                                                    |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| BIPIA [[paper](https://arxiv.org/abs/2312.14197)]                           | Indirect prompt injection attacks                                        |
| CoDI-Eval [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/29734)] | The capability of LLMs to respond to the constraints in instructions.    |
| CMATH [[paper](https://arxiv.org/abs/2306.16636)]                           | Mathematical problem solving                                             |
| VasE [[paper](https://arxiv.org/abs/2312.03121)]                            | Evaluation through voting theory framework                               |
| Head-to-Tail [[paper](https://arxiv.org/abs/2308.10168)]                    | Assess the ability of LLMs to internalize facts                          |
| REGBENCH [[paper](https://arxiv.org/abs/2401.12973)]                        | In-context Language Learning(ICLL)                                       |
| Freshbench [[paper](https://arxiv.org/abs/2405.08460)]                      | A dynamic assessment framework for temporal generalisation               |
| LongctxBench [[paper](https://arxiv.org/abs/2407.01527)]                    | Efficiency in handling long contexts                                     |
| LongICLBench [[paper](https://arxiv.org/abs/2404.02060)]                    | Benchmark for extreme-label classification in long contexts              |
| LooGLE [[paper](https://arxiv.org/abs/2311.04939)]                          | Benchmark for long-dependency understanding.                             |
| MAGNIFICO [[paper](https://arxiv.org/abs/2310.11634)]                       | Learning new interpretations in-context.                                 |
| MANGO [[paper](https://arxiv.org/abs/2403.19913)]                           | Benchmark for mapping and navigation capabilities.                       |
| PRE [[paper](https://arxiv.org/abs/2401.15641)]                             | Evaluation framework inspired by academic peer review.                   |
| RULER [[paper](https://arxiv.org/abs/2404.06654)]                           | Enhanced benchmark for long-context evaluations extending the NIAH test. |
| S3EVAL [[paper](https://arxiv.org/abs/2310.15147)]                          | Flexible evaluation method using complex synthetic tasks.                |
| SuperCLUE [[paper](https://arxiv.org/abs/2307.15020)]                       | Comprehensive benchmark for Chinese user preferences.                    |
| Zhujiu [[paper](https://arxiv.org/abs/2308.14353)]                          | Multi-dimensional evaluation benchmark for Chinese LLMs.                 |

**Other Evaluations**

1. Is attention required for icl? exploring the relationship between model architecture and in-context learning ability [[paper](https://openreview.net/forum?id=Qwq4cpLtoX)]
2. Revenge of the fallen? recurrent models match transformers at predicting human language comprehension metrics [[paper](https://arxiv.org/abs/2404.19178)]
3. Veliki jezikovni modeli so strojni ucenci v casu sklepanja [[paper](<https://erk.fe.uni-lj.si/2023/papers/grm(veliki_jezikovni).pdf>)]
4. Benchmarking neural decoding backbones towards enhanced on-edge ibci applications [[paper](https://arxiv.org/abs/2406.06626)]
5. Does transformer interpretability transfer to rnns? [[paper](https://arxiv.org/abs/2404.05971)]
6. From words to numbers: Your large language model is secretly a capable regressor when given in-context examples [[paper](https://arxiv.org/abs/2404.07544)]
7. How well can a long sequence model model long sequences? comparing architechtural inductive biases on long-context abilities [[paper](https://arxiv.org/abs/2407.08112)]
8. LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding [[project](https://github.com/Ojiyumm/LongBench_RWKV)]
9. Benchmark RWKV on Long Range Arena [[project](https://github.com/diggerdu/rwkv-long-range-arena)]
10. Benchmark RWKV on MT-Bench [[project](https://github.com/Ojiyumm/MT_BENCH_RWKV)]
11. Compare llms [[project](https://github.com/BiroAd/compare_llms)]

### Others

2. Spiking mixers for robust and energy-efficient vision-and-language learning [[paper](https://openreview.net/forum?id=FyZaVdQLdJ)]
3. Visualrwkv: Exploring recurrent neural networks for visual language models [[paper](https://arxiv.org/abs/2406.13362)]
4. A unified implicit attention formulation for gated-linear recurrent sequence models [[paper](https://arxiv.org/abs/2405.16504)]
5. Modern sequence models in context of multi-agent reinforcement learning [[paper](https://epub.jku.at/obvulioa/urn/urn:nbn:at:at-ubl:1-81083)]
6. Generative calibration for in-context learning [[paper](https://arxiv.org/abs/2310.10266)]
7. MiniRWKV-4 [[project](https://github.com/StarRing2022/MiniRWKV-4)]
8. RAG system for RWKV [[project](https://github.com/AIIRWKV/RWKV-RAG)]
9. Dlip-RWKV [[project](https://github.com/StarRing2022/Dlip-RWKV)]
10. Project to extend the functionality of RWKV LM [[project](https://github.com/yynil/RWKV_LM_EXT)]
11. Reward Model based on RWKV [[project](https://github.com/Mazidad/rwkv-reward-enhanced)]
12. RWKV Twitter Bot Detection Project [[project](https://github.com/Max-SF1/Bot-Ani-RWKV-twitter-bot-detection)]
13. RWKV-PEFT [[project](https://github.com/JL-er/RWKV-PEFT) [project](https://github.com/Seikaijyu/RWKV-PEFT-Simple)]
14. RWKV5-infctxLM [[project](https://github.com/JL-er/RWKV5-infctxLM)]
15. DecisionRWKV [[project](https://github.com/ancorasir/DecisionRWKV)]
16. A 20M RWKV v6 can do nonogram [[project](https://github.com/LeC-Z/RWKV-nonogram)]
17. RWKV for industrial time-series prediction [[project](https://github.com/ShixiangLi/RWKV_RUL)]
18. TrainChatGalRWKV [[project](https://github.com/SynthiaDL/TrainChatGalRWKV)]
19. Tinyrwkv: A tinier port of RWKV-LM [[project](https://github.com/wozeparrot/tinyrwkv)]
20. RWKV LLM servicer for SimpleAI [[project](https://github.com/Nintorac/simple_rwkv)]
21. A RWKV management and startup tool [[project](https://github.com/josStorer/RWKV-Runner)]
22. Llama-node: Node.js Library for Large Language Model [[project](https://github.com/Atome-FE/llama-node)]
23. RWKV godot interface module [[project](https://github.com/harrisonvanderbyl/godot-rwkv)]
24. HF for RWKVRaven Alpaca [[project](https://github.com/StarRing2022/HF-For-RWKVRaven-Alpaca)]
25. HF for RWKVWorld LoraAlpaca [[project](https://github.com/StarRing2022/HF-For-RWKVWorld-LoraAlpaca)]
26. Training a reward model for RLHF using RWKV [[project](https://github.com/jiamingkong/rwkv_reward)]
27. Attempt to use RWKV to achieve infinite context length for decision-making [[project](https://github.com/typoverflow/Decision-RWKV)]
28. Reinforcement Learning Toolkit for RWKV [[project](https://github.com/OpenMOSE/RWKV-LM-RLHF)]
29. State tuning with Orpo of RWKV v6 can be performed with 4-bit quantization [[project](https://github.com/OpenMOSE/RWKV-LM-State-4bit-Orpo)]
30. Fine Tuning RWKV [[project](https://github.com/Durham/RWKV-finetune-script)]
31. LoRA fork of RWKV-LM [[project](https://github.com/if001/RWKV-LM-LoRA-ja)]
32. A Lightweight and Extensible RWKV API for Inference [[project](https://github.com/ssg-qwq/RWKV-Light-API)]
33. RWKV StateTuning [[project](https://github.com/Jellyfish042/RWKV-StateTuning)]
34. Continous batching and parallel acceleration for RWKV6 [[project](https://github.com/00ffcc/chunkRWKV6)]
35. RWKV-LM interpretability research [[project](https://github.com/UnstoppableCurry/RWKV-LM-Interpretability-Research)]
36. Web-RWKV Inspector [[project](https://github.com/cryscan/web-rwkv-inspector)]
37. RWKV-UMAP: Recording and visualizing transitions of RWKV-LM internal states [[project](https://github.com/Prunoideae/rwkv_umap)]
38. A converter and basic tester for rwkv onnx [[project](https://github.com/Dan-wanna-M/rwkv-tensorrt)]
39. Llama and other large language models on iOS and MacOS offline using GGML library [[project](https://github.com/guinmoon/LLMFarm)]
40. Enhancing LangChain prompts to work better with RWKV models [[project](https://github.com/jiamingkong/RWKV_chains)]
41. Visual server for RWKV-Ouroboros project [[project](https://github.com/neromous/RWKV-Ouroboros-app)]
42. A set of bash scripts to automate deployment of RWKV modelswith the use of KoboldCpp on Android - Termux [[project](https://github.com/latestissue/AltaeraAI)]
43. Distill (hybrid) RWKV model with Llama [[project](https://github.com/yynil/RWKVinLLAMA)]
44. GPTQ for RWKV [[project](https://github.com/3outeille/GPTQ-for-RWKV)]
45. Transplant the RWKV-LM to ROCm platform [[project](https://github.com/Alic-Li/RWKV-LM-AMD-Radeon-ROCm-hip)]
46. A deep learning engine that can be built into your video game [[project](https://github.com/SingingRivulet/InnerDNN)]
47. EasyChat Q&A System [[project](https://github.com/Ow1onp/EasyChat-Server)]
48. ChatRWKV PC [[project](https://github.com/mosterwei13/ChatRWKV_PC)]
49. PlantFlower Datasets: Based on the RWKV World model dataset [[project](https://github.com/lovebull/PlantFlowerDatasets)]

## Contributing

We welcome contributions to RWKV-survey! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch with your changes.
3. Submit a pull request with a clear description of your changes.

You can also open an issue if you have anything to add or comment.

## Citation

If you find this project useful in your research or work, please consider citing it:

```
@article{li2024survey,
      title={A Survey of RWKV},
      author={Li, Zhiyuan and Xia, Tingyu and Chang, Yi and Wu, Yuan},
      journal={arXiv preprint arXiv:2412.14847},
      year={2024}
}

```

## Acknowledgements
