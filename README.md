采用双路记忆，一路存储原始对话，一路存储LLM处理过后的对话
![alt text](img/image.png)

![alt text](img/image1.png)

在Agent中会同时对原始记忆和处理过后的记忆进行回答

配置:在.env当中需要填入两个memos的api——key，你需要创建两个项目来分别存储

![alt text](img/image2.png)