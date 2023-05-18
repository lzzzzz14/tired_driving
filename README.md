【赛题名称】
基于端云算力协同的疲劳驾驶智能识别
【赛题背景】
       本次大赛以国家重大战略需求为导向，聚焦数字经济领域中智能驾驶场景（疲劳/分神驾驶检测），利用端侧算力单元与云上算力中心协同，让车辆能够更准确更迅速的检测疲劳/分神驾驶，及时提醒司机，减少交通事故。
       选手要解决的核心难题是如何结合与协同端云算力，研发端云算力协同应用系统，让疲劳/分神检测更迅速。参赛选手可选用搭载OpenHarmony等系统的开发板、小车、摄像头等终端硬件来模拟车辆，并选用云上AI开放能力来使用云侧算力。
       参赛选手需要识别出驾驶过程中的闭眼、哈欠、打电话、左顾右盼等疲劳/分神行为，为了兼顾算法的精度和速度，参赛选手可以协同利用端侧设备和云侧平台的算力。其中初赛阶段的判分在云侧进行，参赛选手必须基于华为云AI开发平台ModelArts开发，提交适配云侧ModelArts的算法包；复赛与决赛阶段的判分在线下进行，参赛选手需要提交适配端侧设备的算法包 + 适配云侧ModelArts的算法包（若为纯端侧方案，则不需要云侧部分），以及方案介绍PPT等。
        疲劳/分神驾驶行为的识别，会综合利用到计算机视觉领域的目标检测、关键点检测、图像分类等基础技术；为提升识别效率，也需要用到剪枝、量化、蒸馏、batch推理等模型方面的优化，以及异步、多线程、pipeline并行等应用层面的优化。因此，本次比赛会综合考察参赛选手的基础理论掌控、模型设计优化、应用开发优化等方面的能力。参赛选手可选用搭载OpenHarmony等系统的开发板、小车、摄像头等终端硬件来模拟车辆。
       华为云大赛官网提供赛题相关的边缘计算、边缘协同、人工智能模链接型训练、鸿蒙开发等技术文档及学习课程材料，助力参赛选手学习相关技术，了解实践操作。点击前往查看
【赛题说明】
 大赛组委会将为参赛选手提供数据集（标注了行为类别的视频片段），选手需要根据自己选择的技术路线进行更精细的标注，再进行相应模型的训练和应用的开发。测试集为多段视频，选手提交的算法包需要以视频为单位输出对应的类别，每段视频输出一个结果，格式为json字符串，举例如下：
 {
    "result":
        {
            "category":0,
            "duration":6000
        }
 }
 其中，category为驾驶行为类别编号（见下表的数据说明），duration为算法推理耗时（单位为毫秒）。详情baseline（包含文档以及代码）>>请点击这里<<，供参赛选手参考。
【数据说明】
 训练数据集包含正常驾驶、闭眼、哈欠、打电话、左顾右盼等5种驾驶行为的视频，类别编号与定义如下表：



编号
驾驶行为
定义
0
正常驾驶
驾驶员处于坐姿状态下正常开车
1
闭眼
驾驶员坐姿状态下闭眼，状态持续≥3s
2
哈欠
驾驶员坐姿状态下打哈欠，状态持续≥3s
3
打电话
驾驶员坐姿状态下左/右手举起手机在耳边，状态持续≥3s
4
左顾右盼
驾驶员坐姿状态下向左/向右/向下/向后看，状态持续≥3s，头部偏转角度≥45°
 补充说明：
1. 画面中驾驶员位于主体位置，面部无遮挡，上半身可见，能够看到双臂，可拍到副驾与后排（即画面中可能有多人）；
2. 正常驾驶状态下，也可能会有短时的闭眼、哈欠、打电话、左顾右盼等行为，但每个动作持续时间＜3s；
3. 打哈欠动作能看到嘴部张开，不会被手或者其他物体遮挡，打哈欠过程中如果出现闭眼，请判为打哈欠，不可判为闭眼；
4. 每段视频时长为5~15秒，一段视频中原则上只会包含一种驾驶行为。
样例数据已经发布在AI Gallery页面，欢迎查阅。
完整训练数据集已发布，点击此处查看。
【提交说明】
一、初赛阶段
 1、提交时间：初赛提交时间段为2023年5月5日09:00 ~ 6月15日18:00，每支队伍每天有1次提交机会，提交结果得分可见。将在7月3日公布初赛50强晋级复赛。
 2、初赛提交方法：在ModelArts左侧导航栏中选择“AI应用管理”，单击模型名称前方箭头，然后单击右侧操作栏中的“发布”，在“发布至AI大赛”选择比赛项目、比赛阶段，单击“提交作品”即将模型提交至评分系统进行判分。
 说明：
 （1）模型提交判分后需等待一定时间（判分系统运行时长与参赛选手提交的模型有关），判分系统完成判分后，可在大赛平台“提交作品”页面查看本作品得分。
 （2）提交模型后请勿在AI Gallery中下架模型，否则会导致判分失败。
补充说明：《申报书》和《报名信息统计表》提交方式
 1、WORD版本/PDF版本提交方式：
 各团队须提交《XX学校第十八届“挑战杯”竞赛“揭榜挂帅”专项赛作品申报书》和《XX学校第十八届“挑战杯”竞赛“揭榜挂帅”专项赛报名信息统计表》两个表格（WORD版可点击下载），应同时提交两个表格的WORD版本和PDF版本到wotugaoxiao@huawei.com。
 其中PDF版本须在WORD版本按要求填写完整真实信息完毕后打印出纸质版，并在指定位置加盖红章后扫描生成，PDF和WORD 版本除盖章外，其余所有信息应保持完全一致。
 2、表格纸质版提交方式：
 附件2报名信息统计表和附件3申报书均需要在6月15日18点前1式2份邮寄到北京市海淀区地锦路1号院K区CD座前台，收件人：张老师，收件电话：13753491670。纸质版如果未按时提交，参赛视为无效。
二、复赛与决赛阶段
初赛前50强入围复赛，复赛作品提交时间段为2023年7月10日09:00 ~ 8月3日18:00；每个团队限提交3次，以最新版为准。并参与8月22日及8月23日的线上视频答辩，8月31日公布复赛获奖名单：5个特等奖，一、二、三等奖各5个。
5个特等奖团队入围擂台赛，擂台赛将于10月份在贵州大学现场举办。预计8月份公布具体擂台赛具体安排。
提交要求：
1、PPT作品方案介绍（PPT方案中请提供相关的举证演示视频及截图等材料）
2、作品代码
统一为ZIP/RAR压缩包，上传到华为云赛道大赛平台<提交作品>。请将压缩包命名为：院校名称+队长姓名+队长手机号+队伍名称（例如：某某大学+某某+137xxxxxxxx+中华战队），作品大小不能超过500MB。