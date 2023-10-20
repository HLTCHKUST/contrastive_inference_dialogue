SUBSEQ_EVENT_TEMPLATE = '''Context:
<speaker1> Hi, Peter, haven't seen you for ages .
<speaker2> Hi, Cathy . I've been busy with my book .
<speaker1> Haven't finished yet ?
<speaker2> I will have in a few days .
<speaker1> Are you going to advertise it yourself ? 
<speaker2> Hmm , some friends suggested I should , but I'm still in two minds about it . 
<speaker1> If I were you , I would . 
<speaker2> Thank you for your advice . I think I'll market it .

Target:
I will have in a few days .
Question:
What subsequent event happens or could happen following the target?
Answer:
Peter informed cathy that his book is almost completed.
###END###
Context:
<speaker1> Hello , this is 5735647 5 . 
<speaker2> Hello , this is John . I wanna speak to Linda , please . 
<speaker1> This is Linda .
<speaker2> Hi , Linda . I want to invite you to a dinner tomorrow evening .

Target:
Hi , Linda . I want to invite you to a dinner tomorrow evening .
Question:
What subsequent event happens or could happen following the target?
Answer:
Linda expresses her confirmation for the dinner with john.
###END###
Context:
<speaker1> Can I still catch T107 for Xiamen ? 
<speaker2> Sorry , sir . The train has already left . 
<speaker1> That's too bad . Can I take another train ? 
<speaker2> Yes . Your ticket is valid for three days . 
<speaker1> That's great ! I will take the next train . 
<speaker2> You have to have your ticket checked . 
<speaker1> Shall I pay extra charge ? 
<speaker2> No extra charge at all . But your berth will be invalid .

Target:
That's great ! I will take the next train .
Question:
What subsequent event happens or could happen following the target?
Answer:
The speaker asks the listener about the arrival time and platform number of the next train.
###END###
Context:
<speaker1> It's lucky that we rode our bike here instead of driving. 
<speaker2> It's a good job that we got here early. Look at all those cars there. They'll never get in. 
<speaker1> You'd better follow me closely. I don't want to lose you. 
<speaker2> Don't worry. I'll keep up. 
<speaker1> We go in over there. Gate B. Peter said they're pretty good tickets. 
<speaker2> Where are they? 
<speaker1> They're right behind the goal. 
<speaker2> Oh,do we have to stand up all the time? 
<speaker1> That's right. 
<speaker2> I hope we can see the match clearly. 
<speaker1> That's why we've come early. The earlier, the better.

Target:
That's why we've come early. The earlier, the better.
Question:
What subsequent event happens or could happen following the target?
Answer:
They got front seats in the stadium and enjoyed the match.
###END###
Context:
<speaker1> How ' s it going ? 
<speaker2> I ' m great . Thanks . 
<speaker1> What do you need ? 
<speaker2> I need to know if I have any fees to pay . 
<speaker1> Actually , you do owe some fees . 
<speaker2> How much do I owe ? 
<speaker1> Your fees total $ 235.13 . 
<speaker2> That ' s crazy ! 
<speaker1> You need to pay these fees soon .

Target:
You need to pay these fees soon .
Question:
What subsequent event happens or could happen following the target?
Answer:
The speaker suggested the listener to pay the late fee immediately otherwise speaker's library pass would get rejected.
###END###
'''

CAUSE_TEMPLATE = '''Context:
<speaker2> hello . 
<speaker2> holiday inn . 
<speaker2> may help you ? 
<speaker1> yes , i 'd like to book a room for 2 on the seventh of june . 
<speaker2> ok. let me check . 
<speaker2> well , would you like a smoking or non-smoking room ? 
<speaker1> well , how much is the non-smoking room ? 
<speaker2> $ 80 , plus the 10 per cent room tax . 
<speaker1> ok , that 'll be fine .

Target:
ok , that 'll be fine .
Question:
What is or could be the cause of target?
Answer:
The speaker finds the rate of the room affordable.
###END###
Context:
<speaker1> How's your brother doing ? 
<speaker2> As a matter of fact, he hasn't been feeling too well .
<speaker1> I'm sorry to hear that . What's the matter ? 
<speaker2> Tell him I hope he 's better soon . 
<speaker1> I'll tell him . Thanks for asking about him .

Target:
As a matter of fact, he hasn't been feeling too well.
Question:
What is or could be the cause of target?
Answer:
The speaker's brother was suffering from illness.
###END###
Context:
<speaker2> hi . 
<speaker2> how can we help you today ? 
<speaker1> yeah , i 'd like to get my hair cut a little . 
<speaker2> well , can we interest you in today 's special ? 
<speaker1> um ... nah , nah . 
<speaker2> we 'll shampoo , cut and style your hair for one unbelievable low price of $ 9.99 . 
<speaker2> plus , we 'll give you a clean shaved to help you relax . 
<speaker1> i just want to get my hair cut . 
<speaker1> a little of the top and sides . 
<speaker1> that 's all .

Target:
plus , we 'll give you a clean shaved to help you relax .
Question:
What is or could be the cause of target?
Answer:
There is a offer going on the saloon for all its customer.
###END###
Context:
<speaker1> Waiter , a menu please ! 
<speaker2> Here you are . 
<speaker1> Thank you . Could you tell me the specials today ? 
<speaker2> The special today is fried chicken , and beef is good too . 
<speaker1> Ok , let's think about it for a minute . 
<speaker2> Well , I'll be back in a minute .

Target:
Ok , let's think about it for a minute .
Question:
What is or could be the cause of target?
Answer:
The speaker has come to a restaurant to have dinner.
###END###
Context:
<speaker1> Is the room ready for the meeting , Miss Chen ? 
<speaker2> Yes , Mr . Li . 
<speaker1> How about the microphone and speakers ? 
<speaker2> I also have done it . 
<speaker1> Good . Have you prepared some paper and pencils for the participants ? 
<speaker2> Yes . They have been laid by their name cards on the meeting table for each attendant .

Target:
Yes . They have been laid by their name cards on the meeting table for each attendant .
Question:
What is or could be the cause of target?
Answer:
Each attendent of the meeting needs paper and pen for making notes while discussing important issues.
###END###
'''

PREREQUISITE_TEMPLATE = '''Context:
speaker1> I'm back . 
<speaker2> What have you done ? 
<speaker1> Going shopping. It tires me too much . 
<speaker2> Why don't you go shopping online ? 
<speaker1> Can I ?
<speaker2> Why not ? Let me recommend a website . 
<speaker1> OK. What does it sell ? 
<speaker2> It sells almost everything you can see in the department . 
<speaker1> That's great .

Target:
Why not ? Let me recommend a website . 
Question:
What is or could be the prerequisite of target?
Answer:
There are plenty of e-commerce websites available online that sell multiple varieties of stuff.
###END###
Context:
<speaker1> what 's that noise ? 
<speaker2> it 's my chicken . 
<speaker2> she sounds like that every time she lays eggs . 
<speaker1> fresh eggs for breakfast . 
<speaker1> i 'll bring the bacon . 
<speaker2> no hurry . 
<speaker2> so far , she 's only laid one egg .

Target:
so far , she 's only laid one egg .
Question:
What is or could be the prerequisite of target?
Answer:
The speaker used to have his chicken's eggs for the breakfast.
###END###
Context:
<speaker2> how about going to the airport by car ? 
<speaker1> it 'll take ages to park . 
<speaker1> let 's take the bus . 
<speaker2> the bus is too slow . 
<speaker2> we have to take the train . 
<speaker2> it only takes 45 minutes . 
<speaker1> ok then .

Target:
it 'll take ages to park .
Question:
What is or could be the prerequisite of target?
Answer:
The parking space is not available at the airport.
###END###
Context:
<speaker1> What's wrong with you, sir? 
<speaker2> I've got a headache and a slight fever. Besides, I cough badly. 
<speaker1> How long have you been like this? 
<speaker2> Three days. <speaker1> Let me take your temperature.

Target:
I've got a headache and a slight fever. Besides, I cough badly.
Question:
What is or could be the prerequisite of target?
Answer:
The doctor examines the speaker for various symptoms.
###END###
Context:
<speaker1> Hello , miss . Check out please . 
<speaker2> OK , may I have your key ? 
<speaker1> Sure , here you are . 
<speaker2> Was everything satisfactory ? 
<speaker1> Yes . I enjoy my stay here . 
<speaker2> You have stayed here for 3 nights , that's $ 230 . 
<speaker1> Here is the money .

Target:
Hello , miss . Check out please .
Question:
What is or could be the prerequisite of target?
Answer:
The speaker booking time is over now.
###END###
'''

REACTION_TEMPLATE = '''Context:
<speaker1> Are you doing anything tonight ? 
<speaker2> No , nothing . Why ? 
<speaker1> Do you like western music ? 
<speaker2> Yes , I do , very much . 
<speaker1> There is a concert tonight . Would you like to go and listen to it ? 
<speaker2> Oh , yes . I'd love to .

Target:
Oh , yes . I'd love to .
Question:
What is the possible emotional reaction of the listener in response to target?
Answer:
The listener is excited to go with the speaker to the concert.
###END###
Context:
<speaker1> Excuse me , sir . 
<speaker2> Yes ? 
<speaker1> Is this your car ? 
<speaker2> Yes , it is . 
<speaker1> I'm afraid you've parked on a double yellow line , sir . 
<speaker2> Good heavens , am I really ? I'm so sorry , I didn't notice . 
<speaker1> I'm sorry , sir , but I'll have to give you a ticket . 
<speaker2> I see . 
<speaker1> May I have your name , please , sir ?

Target:
May I have your name , please , sir ?
Question:
What is the possible emotional reaction of the listener in response to target?
Answer:
The listener feels upset for getting a ticket for parking in the wrong line.
###END###
Context:
<speaker2> i 've decided to get rid of some of my old shoes . 
<speaker2> do you have any suggestions ? 
<speaker1> yes . 
<speaker1> have you planned to throw them away ? 
<speaker1> why not sell them online ?

Target:
why not sell them online ?
Question:
What is the possible emotional reaction of the listener in response to target?
Answer:
The listener agreed with the speaker's suggestion.
###END###
Context:
<speaker1> This is a very good meeting , Liz . 
<speaker2> I am happy that we ' ve finally cleared up some problems . 
<speaker1> I think we have . Is there anything else to discuss ? 
<speaker2> No . That's all , I guess . 
<speaker1> Then let's call it a day , shall we ? 
<speaker2> All right . See you later . 
<speaker1> After a while .

Target:
After a while .
Question:
What is the possible emotional reaction of the listener in response to target?
Answer:
Liz felt satisfied and relieved after the meeting.
###END###
Context:
<speaker1> Peter , have you seen my purse ? 
<speaker2> No , mom . I haven't seen it . 
<speaker1> That's strange . It should be on the desk . 
<speaker2> Mom , did you try the basket on your bicycle ? 
<speaker1> Not yet . Let me see .

Target:
Not yet . Let me see .
Question:
What is the possible emotional reaction of the listener in response to target?
Answer:
The listener was getting anxious.
###END###
'''

MOTIVATION_TEMPLATE = '''Context:
<speaker1> Can you show me how to use chopsticks ? 
<speaker2> With pleasure . 
<speaker1> Oh , it is not easy to learn ! 
<speaker2> I think you are a quick learner . 
<speaker1> Well , I don't think I can manage with it . 
<speaker2> In that case , shall I ask the waiter to bring you a knife and fork ? 
<speaker1> That's good , thank you .

Target:
In that case , shall I ask the waiter to bring you a knife and fork ? 
Question:
What is or could be the motivation of target?
Answer:
The speaker do not want to try the chopsticks anymore.
###END###
Context:
<speaker1> Hey , what's good with you ? 
<speaker2> Not a lot . What about you ? 
<speaker1> I'm throwing a party on Friday . 
<speaker2> That sounds like fun . 
<speaker1> Do you think you can come ? 
<speaker2> I'm sorry . I'm already doing something this Friday . 
<speaker1> What are you going to be doing ? 
<speaker2> My family and I are going to dinner .
<speaker1> I was hoping you would come . 
<speaker2> I'll definitely try to make it the next time . 
<speaker1> I'd better see you there . 
<speaker2> All right . I'll see you next time .

Target:
My family and I are going to dinner .
Question:
What is or could be the motivation of target?
Answer:
The listener wants to take his family out for dinner to spend some good quality time together.
###END###
Context:
<speaker1> I had a busy morning . 
<speaker2> What did you do ? 
<speaker1> I watered all the plants . 
<speaker2> You have a lot of plants . 
<speaker1> Then I did my laundry . 
<speaker2> That takes some time . 
<speaker1> I took the dog for a walk . 
<speaker2> I'll bet he enjoyed his walk . 
<speaker1> I vacuumed the entire house . 
<speaker2> That's a lot of work . 
<speaker1> And then I made lunch . 
<speaker2> I'll bet you were hungry !

Target:
I took the dog for a walk .
Question:
What is or could be the motivation of target?
Answer:
The speaker is an animal lover.
###END###
Context:
<speaker1> Do you have desserts ? 
<speaker2> Yes , we have many kinds of pies and puddings . 
<speaker1> Can you give us more information ? 
<speaker2> Sure . We have lemon pies , apple pies , coffee rum pies and ice cream . 
<speaker1> I would like to have the coffee rum pies 

Target:
I would like to have the coffee rum pies .
Question:
What is or could be the motivation of target?
Answer:
The speaker wants to try his favorite coffee rum pies at this new bakery shop.
###END###
Context:
<speaker1> Excuse me , I wonder if you could help me ? 
<speaker2> Of course , what can I do for you ? 
<speaker1> Well , I hate to have to say this , but I'm not happy with my room . 
<speaker2> Oh , what exactly is problem ? 
<speaker1> Well , the traffic is very loud . I got no sleep last night . 
<speaker2> Oh , I'm so sorry , Sir . I'll see what I can do about that .

Target:
Oh , I'm so sorry , Sir . I'll see what I can do about that .
Question:
What is or could be the motivation of target?
Answer:
The speaker assures the listener that he will sort his problem out as soon as possible.
###END###
'''