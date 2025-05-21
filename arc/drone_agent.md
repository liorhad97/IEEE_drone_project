in the Drone Agent 
we will use VIT for images
and for scalaer we will juse bert encoder in order to represent the state with Attention machanizem 
there is a  problome how do i combine image data with scalar data 
like its a big prolome
as well the Agent should look on 2 tensors like the real image and the Lidar sensor image
from detecting unseen objects that can lead to catching like nets or big lives that are not very visible as well as water 
we will use transformer based Reancforcment Learining using DQN and Replay buffer of the state

one sulotion is to 
try too make a make a reperesntation of the 
+----------------+    +----------------+    +----------------+
| Camera Image   |    | LIDAR Image    |    | Scalar Data    |
+----------------+    +----------------+    +----------------+
        |                     |                     |
        v                     v                     v
+----------------+    +----------------+    +----------------+
| ViT_Camera     |    | ViT_LIDAR      |    | ScalarEmbedding|
+----------------+    +----------------+    +----------------+
        |                     |                     |
        v                     v                     v
+----------------+    +----------------+    +----------------+
| Camera Tokens  |    | LIDAR Tokens   |    | Scalar Token   |
+----------------+    +----------------+    +----------------+
        |                     |                     |
        +---------+-----------+----------+----------+
                  |                              |
                  v                              v
          +----------------+              +----------------+
          | CLS Token      |<-------------| Combined Tokens|
          +----------------+              +----------------+
                  |                              |
                  +---------+--------------------+
                            |
                            v
                  +--------------------+
                  | TransformerEncoder |
                  +--------------------+
                            |
                            v
                  +-----------------+
                  | CLS Output      |
                  +-----------------+
                            |
                            v
                  +----------------+
                  | Q_Network      |
                  +----------------+
                            |
                            v
                  +----------------+
                  | Q-Values       |
                  +----------------+
