#语音助手是这样子的（二）
前一节我们介绍了语音助手的基本框架与核心技术，本节我们将介绍使用[OpenDial](http://www.opendial-toolkit.net/)来设计对话的管理与流程
##再来看场景
前一节中我们给出了一个设置闹钟的场景，现在让我们重新使用OpenDial的视角来看一下这个场景
> 用户：设置闹钟<br/>
> Siri：请问您需要设置几点的闹钟？<br/>
> 用户：明天早上六点的<br/>
> Siri：好的，已经把闹钟设置到了明天早上六点<br/>

在这个场景中，如果我们稍作抽象则可以这样来看，“设置闹钟”可以抽象为一个动作叫做SetAlarm，而这个动作是来自用户的，我们定义其为a\_u（*action of user*）,而“设置闹钟”是这个动作的一种语言表达u\_u（*utterance of user*），语言的多样性允许用户使用其他的表述如“请为我设置一个闹钟”等等。同样，Siri的答复和操作也可以抽象，系统的答复“请问您需要设置几点的闹钟？”是u\_m（*utterance of machine*）的一种语言表达，而询问设置闹钟的具体时间可以抽象为一个叫做RequestTime的动作，即a\_m（*action of machine*）是RequestTime。<br/>
如上的这些抽象其实是按照OpenDial的规范进行描述的，那么接下来我们就来研究如何把这样的一个对话场景设置到OpenDial中。<br/>
##OpenDial
[OpenDial](http://www.opendial-toolkit.net/)是一个使用概率规则与贝叶斯网络实现的开源对对话系统引擎。读者可以参考[OpenDial用户手册](http://www.opendial-toolkit.net/user-manual)中的介绍完整地学习OpenDial的使用，这里我们只针对本文中的场景介绍其简单的使用与配置。<br/>
首先，启动OpenDial的可视化工具（.\scripts\opendial.bat），新建一个领域

