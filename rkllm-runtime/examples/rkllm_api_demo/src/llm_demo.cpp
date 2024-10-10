// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string.h>
#include <unistd.h>
#include <string>
#include "rkllm.h"
#include <fstream>
#include <iostream>
#include <csignal>
#include <vector>

#define PROMPT_TEXT_PREFIX "<|im_start|>system You are a helpful assistant. <|im_end|> <|im_start|>user"
#define PROMPT_TEXT_POSTFIX "<|im_end|><|im_start|>assistant"


using namespace std;
LLMHandle llmHandle = nullptr;

void exit_handler(int signal)
{
    if (llmHandle != nullptr)
    {
        {
            cout << "程序即将退出" << endl;
            LLMHandle _tmp = llmHandle;
            llmHandle = nullptr;
            rkllm_destroy(_tmp);
        }
    }
    exit(signal);
}

void callback(RKLLMResult *result, void *userdata, LLMCallState state)
{
    if (state == RKLLM_RUN_FINISH)
    {
        printf("\n");
    } else if (state == RKLLM_RUN_ERROR) {
        printf("\\run error\n");
    } else if (state == RKLLM_RUN_GET_LAST_HIDDEN_LAYER) {
        /* ================================================================================================================
        若使用GET_LAST_HIDDEN_LAYER功能,callback接口会回传内存指针:last_hidden_layer,token数量:num_tokens与隐藏层大小:embd_size
        通过这三个参数可以取得last_hidden_layer中的数据
        注:需要在当前callback中获取,若未及时获取,下一次callback会将该指针释放
        ===============================================================================================================*/
        if (result->last_hidden_layer.embd_size != 0 && result->last_hidden_layer.num_tokens != 0) {
            int data_size = result->last_hidden_layer.embd_size * result->last_hidden_layer.num_tokens * sizeof(float);
            printf("\ndata_size:%d",data_size);
            std::ofstream outFile("last_hidden_layer.bin", std::ios::binary);
            if (outFile.is_open()) {
                outFile.write(reinterpret_cast<const char*>(result->last_hidden_layer.hidden_states), data_size);
                outFile.close();
                std::cout << "Data saved to output.bin successfully!" << std::endl;
            } else {
                std::cerr << "Failed to open the file for writing!" << std::endl;
            }
        }
    } else if (state == RKLLM_RUN_NORMAL) {
        printf("%s", result->text);
        // for(int i=0; i<result->num; i++)
        // {
        //     printf("%d token_id: %d logprob: %f\n", i, result->tokens[i].id, result->tokens[i].logprob);
        // }
    }
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " model_path max_new_tokens max_context_len\n";
        return 1;
    }

    signal(SIGINT, exit_handler);
    printf("rkllm init start\n");

    //设置参数及初始化
    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = argv[1];
    param.top_k = 1;
    param.max_new_tokens = std::atoi(argv[2]);
    param.max_context_len = std::atoi(argv[3]);

    param.skip_special_token = true;
    param.extend_param.base_domain_id = 0;
    int ret = rkllm_init(&llmHandle, &param, callback);
    if (ret == 0){
        printf("rkllm init success\n");
    } else {
        printf("rkllm init failed\n");
        exit_handler(-1);
    }

    vector<string> pre_input;
    pre_input.push_back("Summarize the text in 50 words: And the number one lesson I could offer you where your work is concerned is this. Become so skilled, so vigilant, so flat, out, fantastic at what you do that your talent cannot be dismissed. When I was a kid, I thought I was going to be a social worker or be a teacher or somehow be in an environment where I would be connected and speaking to people to honor that calling. I had no idea that God could dream this big dream of television life for myself. But everybody has a calling. and your real job in life is to figure out as soon as possible what that is, who you were meant to be and begin to honor that in the best way possible for yourself. We have seen over the years on the Oprah Show so many people who've been able to rise to success in their life and only able to rise to success because they answered the call. Lots of people think that it's about being famous or about being known or about doing big, big, incredible things. What you're going to hear today and see through the stories is that sometimes the calling is right in your own neighborhood. Sometimes the calling is something that was just a whisper to you. And when you begin to honor that whisper and to follow that, you'd end up being the best that you could be. I actually thought at the time that being thin made me better. My identification with form, my wanting so desperately to be in a size 10 gene was so prominent identifying with that. That shortly after this, I gained five pounds. And I was invited to a party at the time by Don Johnson, a member of Dreamy Don Johnson. I was invited to a holiday party. And I did not go to that party because I thought the five pounds made me too fat, not good enough to be a Don Johnson's party. Sounds kind of sick. Now I know. But that's what the ego does. It is, it is sick. It's wily, it's cunning, it's deceptive. It's an imposter imposing on the real you, making you often think that you're something that you're not. You're not the shape of your body. You're not your status. You're not your position in life. You're not the car you drive, no matter how fancy it is. You're not your house or your square footage. For all of you watching right now, here's another way to better understand ego. What I'm talking about, Eckhart taught me this. Everybody try this from home or wherever you are right now. Close your eyes and notice what you're thinking. Notice your thoughts. Now, who is thinking those thoughts? Can you notice that there is a space where you are observing the thoughts and where you are aware of the thoughts that you are observing? Eckhart says, you are that awareness disguised as a person. And when I got that, I understood the difference between my true self and my ego self. Over the years, we've heard invaluable information that in a moment of danger could be life-saving. Nothing I heard made a greater impression on me than back in 1991. I think we were beginning our fifth season and man named Sanford Strong came on and shared this. You never forgot it. Rule number one. And frankly, it's probably in my opinion the most important. Never allow them to take you somewhere else. Never. If everyone in this room and everyone watching this program has never drawn the line and made a decision on crime protection, you better make it when they decide to move you from crime scene number one to crime scene number two. Because the crime scene number two, it's going to be isolated, you won't choose it, you'll be the focus of the crime. I think what was so interesting about that piece of advice is at the time that we all heard that we had been trained as women people especially to believe that you just do whatever they say, whatever they say, go along with it. And what's important for those of you who are watching right now and Those of you who are on the internet with us right now to know is that it's all about using your gut in the moment and people who've survived horrific circumstances talk about listening to that intuition and every move being made calculated on listening to what your gut says. What the experts now say is do not allow yourself to be taken to the second location because anybody who is trying to harm you wants to get you to an isolated place where they can do that without other people seeing or knowing it. So in that moment of making the decision, oh, you're going to shoot me, if you're going to shoot me, You have to shoot me now rather than shoot me in isolation where nobody can see you. Whenever somebody has said something that made the little hairs on my arms or neck stand up, I know that if that's happening to me, that's also happening to somebody else too. The way to make movies is to do stuff that you love because, you know, for 25 years, I worked on the Oprah Show and Stemmin will tell you that there were nights that I came home and I almost, you know, it was hard to even take off my clothes because I knew I was gonna be getting up for hours later. But I never really felt exhausted, like, I felt exhausted, but I never felt depleted. So do the work that comes straight from the soul of you. from your background, from stories that you've grown up with, from stories that bring you passion, from stories that you not just urine to tell, but that if you don't tell them they won't get told. And when you are operating, you know, the single greatest wisdom, I think I've ever received other than when people show you they are, is that the key to full filament, success, happiness, contentment in life is when you align your personality with what your soul actually came to do. I believe everybody has a soul and has their own personal spiritual energy. So when you can use your personality to serve whatever that thing is, you can't help but be successful. So if you do films that come from the interior of your soul You do work, you do art that comes from the interior of you. You cannot miss. It's only when you're doing stuff that you think might make money. You think it may be a hit or you think it may bring you some level of attention or success. That isn't what does it. I would have to say that all of the great wonderful experiences of my life that have brought me to this moment have come from working from the interior of myself. And so that's why it feels so authentic because it actually is. So when you do that, you'll win. And everybody has a different talent. And the reason we're all so messed up is because you're looking at everybody else's talent. and wishing you had some of their talent. All the energy that you spend thinking about, wishing about being jealous of, invious of anybody else, is energy that you're not only putting out just going to come back to you negatively, but you're taking that away from you. All your energy should be forced on what do I have to offer, what do I I have to give how can I be used in service because Dr. King's message of not everybody can be famous but everybody can be great because greatness is determined by service and there is not a job in here that you can do that you don't switch the paradigm to service and not make that job more fulfilling. I don't care what the job is. If you say, I'm a singer, I'm a dancer, I'm an artist, I'm a teacher, I'm a nurse, I'm a doctor, I'm a janitor, I'm a clerk, if you say, if I look at this from, how do I use this in service to something bigger than myself? no longer becomes a job. It becomes an offering to the world. I first started making money and it was, you know, my salary or my earnings were published all over the place.");
    pre_input.push_back("他不多久就睡熟了，梦见小时候见到的非洲，长长的金色海滩和白色海滩，白得刺眼，还有高耸的海岬和褐色的大山。他如今每天夜里都神游那道海岸，在梦中听见拍岸海浪的隆隆声，看见土人驾船穿浪而行。他睡着时闻到甲板上柏油和填絮的气味，还闻到早晨陆地上刮来的微风带来的非洲气息。通常一闻到陆地上刮来的微风，他就醒来，穿上衣服去叫醒那男孩。然而今夜陆地上刮来的微风的气息来得很早，他在梦中知道时间尚早，就继续把梦做下去，看见群岛间那些白色浪峰从海面上升起，随后梦见了加那利群岛的各个港湾和锚泊地。他不再梦见风暴，不再梦见妇女们，不再梦见发生过的大事，不再梦见大鱼，不再梦见打架，不再梦见角力，不再梦见他的妻子。他如今只梦见某些地方和海滩上的狮子。它们在暮色中像小猫一般嬉耍着，他爱它们，如同爱这男孩一样。他从没梦见过这男孩。他就这么醒过来，望望敞开的门外边的月亮，摊开长裤穿上。他在窝棚外撒了尿，然后顺着大路走去叫醒男孩。他被清晨的寒气弄得直哆嗦。但他知道哆嗦了一阵后会感到暖和，要不了多久就要去划船了。男孩住的那所房子的门没有上锁，他推开门，光着脚悄悄走进去。男孩在外间一张帆布床上熟睡着，老人靠着外面射进来的残月的光线，清楚地看见他。他轻轻握住男孩的一只脚，直到男孩醒来，转过脸来对他望着。老人点点头，男孩从床边椅子上拿起他的长裤，坐在床沿上穿裤子。老人走出门去，男孩跟在他背后。他还是昏昏欲睡，老人伸出胳臂搂住他的肩膀说，“对不起。”“哪里，”男孩说。“男子汉就该这么干。”他们顺着大路朝老人的窝棚走去，一路上，有些光着脚的男人在黑暗中走动，扛着他们船上的桅杆。他们走进老人的窝棚，男孩拿起装在篮子里的钓索卷儿，还有鱼叉和鱼钩，老人把绕着帆的桅杆扛在肩上。“我们把家什放在船里，然后喝一点吧。”他们在一家清早就营业的供应渔夫的小吃馆里，喝着盛在炼乳听里的咖啡。“你睡得怎么样，老大爷？”男孩问。他如今清醒过来了，尽管要他完全摆脱睡魔还不大容易。“睡得很好，马诺林，”老人说。“我感到今天挺有把握。”“我也这样，”男孩说。“现在我该去拿你我用的沙丁鱼，还有给你的新鲜鱼饵。那条船上的家什总是他自己拿的。他从来不要别人帮他拿东西。”“我们可不同，”老人说。“你还只五岁时我就让你帮忙拿东西来着。”“我记得，”男孩说。“我马上回来。再要杯咖啡吧。我们在这儿可以挂账。”他走了，光着脚在珊瑚石砌的走道上向保藏鱼饵的冷藏所走去。老人慢腾腾地喝着咖啡。这是他今儿一整天的饮食，他知道应该把它喝了。好久以来，吃饭使他感到厌烦，因此从来不带午饭。他在小帆船的船头上放着一瓶水，一整天只需要这个就够了。男孩这时带着沙丁鱼和两份包在报纸里的鱼饵回来了，他们就顺着小径走向小帆船，感到脚下的沙地里嵌着鹅卵石，他们抬起小帆船，让它溜进水里。“祝你好运，老大爷。”“祝你好运，”老人说。他把桨上的绳圈套在桨座的钉子上，身子朝前冲，抵消桨片在水中所遇到的阻力，在黑暗中动手划出港去。其他那些海滩上也有其他船只在出海，老人听到他们的桨落水和划动的声音，尽管此刻月亮已掉到了山背后，他还看不清他们。偶尔有条船上有人在说话。但是除了桨声外，大多数船只都寂静无声。它们一出港口就分散开来，每一条驶向指望能找到鱼的那片海面。老人知道自己要驶向远方，所以把陆地的气息抛在后方，划进海洋上清晨的清新气息中。他划到海里的某一片水域，看见果囊马尾藻闪出的磷光，渔夫们管这片水域叫“大井”，因为那儿水深突然达到七百英寻，海流冲击在海底深渊的峭壁上，激起了旋涡，种种鱼儿都聚集在那儿。这里集中着海虾和可作鱼饵的小鱼，在那些深不可测的水底洞穴里，有时还有成群的柔鱼，它们在夜间浮到紧靠海面的地方，所有在那儿漫游的鱼类都拿它们当食物。老人在黑暗中感觉到早晨在来临，他划着划着，听见飞鱼出水时的颤抖声，还有它们在黑暗中凌空飞走时挺直的翅膀所发出的咝咝声。他非常喜爱飞鱼，因为它们是他在海洋上的主要朋友。他替鸟儿伤心，尤其是那些柔弱的黑色小燕鸥，它们始终在飞翔，在找食，但几乎从没找到过，于是他想，鸟儿的生活过得比我们的还要艰难，除了那些猛禽和强有力的大鸟。既然海洋这样残暴，为什么像这些海燕那样的鸟儿生来就如此柔弱和纤巧？海洋是仁慈并十分美丽的。然而她能变得这样残暴，又是来得这样突然，而这些飞翔的鸟儿，从空中落下觅食，发出细微的哀鸣，却生来就柔弱得不适宜在海上生活。他每想到海洋，老是称她为la mar，这是人们对海洋抱着好感时用西班牙语对她的称呼。有时候，对海洋抱着好感的人们也说她的坏话，不过说起来总是拿她当女性看待的。有些较年轻的渔夫，用浮标当钓索上的浮子，并且在把鲨鱼肝卖了好多钱后置备了汽艇，都管海洋叫elmar，这是表示男性的说法。他们提起她时，拿她当做一个竞争者或一个去处，甚至当做一个敌人。可是这老人总是拿海洋当做女性，她给人或者不愿给人莫大的恩惠，如果她干出了任性或缺德的事儿来，那是因为她由不得自己。月亮对她起着影响，如同对一个女人那样，他想。他平稳地划着，对他说来并不费劲，因为他好好保持在自己的最高速度以内，而且除了水流偶尔打个旋儿以外，海面是平坦无浪的。他正让海流帮他干三分之一的活儿，这时天渐渐亮了，他发现自己已经划到比预期此刻能达到的地方更远了。我在这海底的深渊上转游了一个礼拜，可是一无作为，他想。今天，我要找到那些鲣鱼和长鳍金枪鱼群在什么地方，说不定会有条大鱼跟它们在一起。不等天色大亮，他就放出一个个鱼饵，让船随着海流漂去。有个鱼饵下沉到四十英寻的深处。第二个在七十五英寻的深处，第三个和第四个分别在一百英寻和一百二十五英寻深的蓝色海水中。每个由新鲜沙丁鱼做的鱼饵都是头朝下的，钓钩的钩身穿进小鱼的身子，给扎好，缝牢，因此钓钩的所有突出部分，弯钩和尖端，都给包在鱼肉里。每条沙丁鱼都用钓钩穿过双眼，这样鱼的身子在突出的钢钩上构成了半个环形。钓钩上就没有哪一部分不会叫一条大鱼觉得喷香而美味的。男孩给了他两条新鲜的小金枪鱼，或者叫做长鳍金枪鱼，它们正像铅锤般挂在那两根最深的钓索上，在另外两根上，他挂上一条蓝色大鱼和一条黄色金银鱼，它们已被使用过，但依然完好，而且还有出色的沙丁鱼给它们添上香味和吸引力。每根钓索都像一支大铅笔那么粗，一端给缠在一根青皮钓竿上，这样，只要鱼在鱼饵上一拉或一碰，就能使钓竿下垂，而每根钓索有两个四十英寻长的卷儿，它们可以牢系在其他备用的卷儿上，这一来，如果用得着的话，一条鱼可以拉出三百多英寻长的钓索。这时老人察看着那三根挑出在小帆船一边的钓竿有没有动静，一边缓缓地划着，使钓索保持上下笔直，停留在适当的水底深处。天相当亮了，太阳随时会升起来。淡淡的太阳从海上升起，老人看见其他的船只，低低地挨着水面，离海岸不远，和海流的方向垂直地展开着。跟着太阳越发明亮了，耀眼的阳光射在水面上，随后太阳从地平线上完全升起，平坦的海面把阳光反射到他眼睛里，使眼睛剧烈地刺痛，因此他不朝太阳看，顾自划着。他俯视水中，注视着那几根一直下垂到黑魆魆的深水里的钓索。他把钓索垂得比任何人更直，这样，在黑魆魆的湾流深处的几个不同的深度，都会有一个鱼饵刚好在他所指望的地方等待着在那儿游动的鱼来吃。别的渔夫让钓索随着海流漂动，有时候钓索在六十英寻的深处，他们却自以为在一百英寻的深处呢。不过，他想，我总是把它们精确地放在适当的地方的。问题只在于我的运气就此不好了。可是谁说得准呢？说不定今天就转运。每一天都是一个新的日子。走运当然更好。不过我情愿做到分毫不差。这样，运气来的时候，你就有所准备了。两小时过去了，太阳如今相应地升得更高了，他朝东望时不再感到那么刺眼了。眼前只看得见三条船，它们显得特别低矮，远在近岸的海面上。我这一辈子，初升的太阳老是刺痛我的眼睛，他想。然而眼睛还是好好的。傍晚时分，我可以直望着太阳，不会有眼前发黑的感觉。阳光的力量在傍晚要强一些。不过在早上它叫人感到眼痛。就在这时，他看见一只黑色的长翅膀军舰鸟在他前方的天空中盘旋飞翔。它倏地斜着后掠的双翅俯冲，然后又盘旋起来。“它逮住什么东西了，”老人说出声来。“它不光是找找罢了。”他慢慢划着，直朝鸟儿盘旋的地方划去。他并不着急，让那些钓索保持着上下笔直的位置。不过他还是挨近了一点儿海流，这样，他依然在用正确的方式捕鱼，尽管他的速度要比他不打算利用鸟儿来指路时来得快。军舰鸟在空中飞得高些了，又盘旋起来，双翅纹丝不动。它随即猛然俯冲下来，老人看见飞鱼从海里跃出，在海面上拚命地掠去。“鲯鳅，”老人说出声来。“大鲯鳅。”他把双桨从桨架上取下，从船头下面拿出一根细钓丝。钓丝上系着一段铁丝导线和一只中号钓钩，他拿一条沙丁鱼挂在上面。他把钓丝从船舷放下水去，将上端紧系在船艄一只拳头螺栓上。跟着他在另一根钓丝上安上鱼饵，把它盘绕着搁在船头的阴影里。他又划起船来，注视着那只此刻正在水面上低低地飞掠的长翅膀黑鸟。他看着看着，那鸟儿又朝下冲，为了俯冲，把翅膀朝后掠，然后猛地展开，追踪着飞鱼，可是没有成效。老人看见那些大鲯鳅追随在脱逃的鱼后面，把海面弄得微微隆起。鲯鳅在飞掠的鱼下面破水而行，只等飞鱼一掉下，就飞快地钻进水里。这群鲯鳅真大啊，他想。它们分布得很广，飞鱼很少脱逃的机会。那只鸟可没有成功的机会。飞鱼对它来说个头太大了，而且又飞得太快。他看着飞鱼一再地从海里冒出来，看着那只鸟儿的一无效果的行动。那群鱼从我附近逃走啦，他想。它们逃得太快，游得太远啦。不过说不定我能逮住一条掉队的，说不定我想望的大鱼就在它们周围转游着。我的大鱼总该在某处地方啊。陆地上空的云块这时像山冈般耸立着，海岸只剩下一长条绿色的线，背后是些灰青色的小山。海水此刻呈深蓝色，深得简直发紫了。他仔细俯视着海水，只见深蓝色的水中穿梭地闪出点点红色的浮游生物，阳光这时在水中变幻出奇异的光彩。他注视着那几根钓索，看见它们一直朝下没入水中看不见的地方，他很高兴看到这么多浮游生物，因为这说明有鱼。太阳此刻升得更高了，阳光在水中变幻出奇异的光彩，说明天气晴朗，而陆地上空的云块的形状也说明了这一点。可是那只鸟儿这时几乎看不见了，水面上没什么东西，只有几摊被太阳晒得发白的黄色马尾藻和一只紧靠着船舷浮动的僧帽水母，它那胶质的浮囊呈紫色，具有一定的外形，闪现出虹彩。它倒向一边，然后竖直了身子。它像个大气泡般高高兴兴地浮动着，那些厉害的紫色长触须在水中拖在身后，长达一码。请使用中文归纳上述文本");
    pre_input.push_back("把下面的现代文翻译成文言文: 到了春风和煦，阳光明媚的时候，湖面平静，没有惊涛骇浪，天色湖光相连，一片碧绿，广阔无际；沙洲上的鸥鸟，时而飞翔，时而停歇，美丽的鱼游来游去，岸上与小洲上的花草，青翠欲滴。");
    pre_input.push_back("以咏梅为题目，帮我写一首古诗，要求包含梅花、白雪等元素。");
    pre_input.push_back("上联: 江边惯看千帆过");
    pre_input.push_back("把这句话翻译成中文: Knowledge can be acquired from many sources. These include books, teachers and practical experience, and each has its own advantages. The knowledge we gain from books and formal education enables us to learn about things that we have no opportunity to experience in daily life. We can also develop our analytical skills and learn how to view and interpret the world around us in different ways. Furthermore, we can learn from the past by reading books. In this way, we won't repeat the mistakes of others and can build on their achievements.");
    pre_input.push_back("把这句话翻译成英文: RK3588是新一代高端处理器，具有高算力、低功耗、超强多媒体、丰富数据接口等特点");
    cout << "\n**********************可输入以下问题对应序号获取回答/或自定义输入********************\n"
         << endl;
    for (int i = 0; i < (int)pre_input.size(); i++)
    {
        cout << "[" << i << "] " << pre_input[i] << endl;
    }
    cout << "\n*************************************************************************\n"
         << endl;

    string text;
    RKLLMInput rkllm_input;

    // RKLLMLoraAdapter lora_adapter;
    // memset(&lora_adapter, 0, sizeof(RKLLMLoraAdapter));
    // lora_adapter.lora_adapter_path = "qwen0.5b_fp16_lora.rkllm";
    // lora_adapter.lora_adapter_name = "test";
    // lora_adapter.scale = 1.0;
    // ret = rkllm_load_lora(llmHandle, &lora_adapter);
    // if (ret != 0) {
    //     printf("\nload lora failed\n");
    // }

    // 加载第二个lora
    // lora_adapter.lora_adapter_path = "Qwen2-0.5B-Instruct-all-rank8-F16-LoRA.gguf";
    // lora_adapter.lora_adapter_name = "knowledge_old";
    // lora_adapter.scale = 1.0;
    // ret = rkllm_load_lora(llmHandle, &lora_adapter);
    // if (ret != 0) {
    //     printf("\nload lora failed\n");
    // }

    // 初始化 infer 参数结构体
    RKLLMInferParam rkllm_infer_params;
    memset(&rkllm_infer_params, 0, sizeof(RKLLMInferParam));  // 将所有内容初始化为 0

    // 1. 初始化并设置 LoRA 参数（如果需要使用 LoRA）
    RKLLMLoraParam lora_params;
    lora_params.lora_adapter_name = "test";  // 指定用于推理的 lora 名称

    // 2. 初始化并设置 Prompt Cache 参数（如果需要使用 prompt cache）
    RKLLMPromptCacheParam prompt_cache_params;
    prompt_cache_params.save_prompt_cache = true;                  // 是否保存 prompt cache
    prompt_cache_params.prompt_cache_path = "./prompt_cache.bin";  // 若需要保存prompt cache, 指定 cache 文件路径
    
    rkllm_infer_params.mode = RKLLM_INFER_GENERATE;
    // rkllm_infer_params.lora_params = &lora_params;
    // rkllm_infer_params.prompt_cache_params = &prompt_cache_params;

    // rkllm_load_prompt_cache(llmHandle, "./prompt_cache.bin");
    while (true)
    {
        std::string input_str;
        printf("\n");
        printf("user: ");
        std::getline(std::cin, input_str);
        if (input_str == "exit")
        {
            break;
        }
        for (int i = 0; i < (int)pre_input.size(); i++)
        {
            if (input_str == to_string(i))
            {
                input_str = pre_input[i];
                cout << input_str << endl;
            }
        }
        // text = PROMPT_TEXT_PREFIX + input_str + PROMPT_TEXT_POSTFIX;
        text = input_str;
        rkllm_input.input_type = RKLLM_INPUT_PROMPT;
        rkllm_input.prompt_input = (char *)text.c_str();
        printf("robot: ");

        // 若要使用普通推理功能,则配置rkllm_infer_mode为RKLLM_INFER_GENERATE或不配置参数
        rkllm_run(llmHandle, &rkllm_input, &rkllm_infer_params, NULL);
        // rkllm_run(llmHandle, &rkllm_input, NULL);
    }
    rkllm_destroy(llmHandle);

    return 0;
}