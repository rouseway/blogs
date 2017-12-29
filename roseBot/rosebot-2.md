# 语音助手是这样子的（二）

前一节我们介绍了语音助手的基本框架与核心技术，本节我们将优先介绍使用[OpenDial](http://www.opendial-toolkit.net/)来设计对话的管理与流程。OpenDial的功能很强大，可以实现NLU、DM以及NLG的所有功能，在官网上也有一系列的教程。在本系列博客中，我们仅仅使用OpenDial来进行对话管理（包括NLG），而把NLU剥离出来单独实现。


## 再来看场景

前一节中我们给出了一个设置闹钟的场景，现在让我们重新使用OpenDial的视角来看一下这个场景。
> 用户：设置闹钟<br/>
> Siri：请问您需要设置几点的闹钟？<br/>
> 用户：明天早上六点的<br/>
> Siri：好的，已经把闹钟设置到了明天早上六点<br/>

在这个场景中，如果我们稍作抽象则可以这样来看，“设置闹钟”可以抽象为一个动作叫做SetAlarm，而这个动作是来自用户的，我们定义其为a\_u（action of user）,而“设置闹钟”是这个动作的一种语言表达u\_u（utterance of user），语言的多样性允许用户使用其他的表述如“请为我设置一个闹钟”等等。同样，Siri的答复和操作也可以抽象，系统的答复“请问您需要设置几点的闹钟？”是u\_m（utterance of machine）的一种语言表达，而询问设置闹钟的具体时间可以抽象为一个叫做RequestTime的动作，即a\_m（action of machine）是RequestTime。<br/>

如上的这些抽象其实是按照OpenDial的规范进行描述的，那么接下来我们就来研究如何把这样的一个对话场景设置到OpenDial中。<br/>


## OpenDial

OpenDial是一个使用概率规则与贝叶斯网络实现的开源对对话系统引擎。读者可以参考[OpenDial用户手册](http://www.opendial-toolkit.net/user-manual)中的介绍完整地学习OpenDial的使用，这里我们只针对本文中的场景介绍其简单的使用与配置。<br/>

首先，启动OpenDial的可视化工具（.\scripts\opendial.bat），新建一个领域“Domian -&gt; New”，文件命名为Alarm.xml，点击保存后在“Domain Editor”的标签页里可以设计对话状态了。<br/>
![OpenDial Example 1](https://raw.githubusercontent.com/rouseway/blogs/master/roseBot/rosebot-1.jpg)

OpenDial的配置文件按如下的规范进行构造，一个场景被认为是一个domain，一个domain中包含多个model，每个model由一个trigger触发，通常我们设计三个model分别对应NLU，DM和NLG。NLU的model由u_u触发，DM的model由a_u触发，NLG的model由a_m触发。所以，按照一次对话是由NLU -&gt; DM -&gt; NLG这样的流程形成的思想，我们很自然地可以想象到u_u通过一定的规则（rule）条件（case）触发NLU model，同时在NLU model中设置a_u变量，这样a_u就可以根据一定的规则条件触发DM model，此时只需再在DM model中设置a_m变量，就可以顺利地到达NLG model，从而把需要响应用户的回复设置到u_m中。所以，样例场景就可以配置为如下的形式：

```
<?xml version="1.0" encoding="GB2312"?>
 <domain>
  <model trigger="u_u">
    <rule>
      <case>
        <condition>
          <if var="u_u" relation="=" value="设置闹钟"/>
        </condition>
        <effect prob="1">
          <set var="a_u" value="SetAlarm"/>
        </effect>
      </case>
      <case>
        <condition>
          <if var="u_u" relation="=" value="明天早上六点的"/>
        </condition>
        <effect prob="1">
          <set var="a_u" value="InformTime"/>
        </effect>
      </case>
    </rule>
  </model>
  <model trigger="a_u">
    <rule>
      <case>
        <condition>
          <if var="a_u" relation="=" value="SetAlarm"/>
        </condition>
        <effect prob="1">
          <set var="a_m" value="RequestTime"/>
        </effect>
      </case>
      <case>
        <condition>
          <if var="a_u" relation="=" value="InformTime"/>
        </condition>
        <effect prob="1">
          <set var="a_m" value="ToDoSetAlarm"/>
        </effect>
      </case>
    </rule>
  </model>
  <model trigger="a_m">
    <rule>
      <case>
        <condition>
          <if var="a_m" relation="=" value="RequestTime"/>
        </condition>
        <effect prob="1">
          <set var="u_m" value="请问您需要设置几点的闹钟？"/>
        </effect>
      </case>
      <case>
        <condition>
          <if var="a_m" relation="=" value="ToDoSetAlarm"/>
        </condition>
        <effect prob="1">
          <set var="u_m" value="好的，已经把闹钟设置到了明天早上六点"/>
        </effect>
      </case>
    </rule>
  </model>
</domain>
```
保存如上的配置，在OpenDial工具中切换到“Interaction”标签页，就可以按照场景中的方式与OpenDial进行对话了。
![OpenDial Example 1](https://raw.githubusercontent.com/rouseway/blogs/master/roseBot/rosebot-2.jpg)

OpenDial还有很多功能，比如支持正则表达式、可以设置变量的概率分布等，在这里我们不多介绍，感兴趣的读者可以参考官方指南。


## 归一化表达

我们说u_u由于语言表达的多样性存在很丰富的实例，我们不可能把所有的表达都枚举到OpenDial的配置中，而且诸如“明天早上六点”这样的时间信息，是需要自动传递到下一轮对话的，而不应该是固定写死在配置文件中的。所以，我们将使用一种称为“槽位填充（Slot Filling）”的思想来解决上述的问题。<br/>

**槽位**是来自于自然语言处理中经常使用的规则模板的一个概念，它是对相同语义概念的表达的一种归一化表示。相对的，如果能**填充**到槽位里的表达就叫做相应的实例化表述。我们直观地看一个例子：<br/>

> 【D:set】【D:clock】&nbsp;&nbsp;&nbsp;=&gt;&nbsp;&nbsp;&nbsp;设置闹钟<br/>
> 【F:time】的&nbsp;&nbsp;&nbsp;=&gt;&nbsp;&nbsp;&nbsp;&nbsp;明天早上六点的

我们用【】括起来的部分就是槽位，【D:set】表示的是可以表达“set”这个含义的词，我们需要收集“设置”、“设定”等同义词并且把这些词都定义为“set”，同理可以理解【D:clock】；【F:time】是可以识别时间的一个函数。那么，显然除了例子中的实例表述可以归一化为上面定义的槽位模板，“设定闹铃”也是可以被归一化为相同的槽位模板，“晚上七点的”也一样。所以，多种表述在这里可以被归一化为用槽位模板表达的形式，这样我们可以一定程度上减少在OpenDial中配置太多样的u_u。反过来，当用户输入的实例被泛化为槽位模板的表达时，就意味着每个槽位被相应的字符串填充了，此时我们取出槽位对应的字符串就实现了对关键信息的提取，那么a_m所要采取的动作就有了依据，而且u_m生成的回复也有了参考。<br/><br/>


本节，我们介绍了借助OpenDial实现对话管理的功能，重点讲解了如何配置该模块的xml文件，最后引入了归一化表达的思想，就是想实现NLU与DM的分离，使OpenDial只专注于DM。下一节，我们将详细介绍实现NLU的方法。
