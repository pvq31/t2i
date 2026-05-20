# python3 agent_text2pkl0.py \
#   --output inference/saved_scenes/test3.31/example_test3.31_2.pkl \
#   --scene-text "In the image, there is a bus in the center. A van is positioned behind the bus, and an suv is located in front-right of the bus. A pigeon is on the ground in front of the bus, a sparrow is at the front-left of the bus, and a crow is flying in the air above the bus."

# python3 agent_check_pkl0.py \
#   --scene-text "In the image, there is a bus in the center. A van is positioned behind the bus, and an suv is located in front-right of the bus. A pigeon is on the ground in front of the bus, a sparrow is at the front-left of the bus, and a crow is flying in the air above the bus." \
#   --scene-pkl inference/saved_scenes/test3.31/example_test3.31_2_fixed.pkl \
#   --output inference/saved_scenes/test3.31/example_test3.31_2_fixed.pkl

# python3 infer20.py \
#   --scene-pkls inference/saved_scenes/test3.31/example_test3.31_2_fixed.pkl \
#   --placeholder-prompt "a photo of PLACEHOLDER, realistic urban documentary style, soft lighting, a slight early-morning freshness in the air, a clean natural composition, clear details, and a calm overall mood."


# demo3
# python3 agent_text2pkl0.py \
#   --output inference/saved_scenes/test3.31/example_test3.31_3.pkl \
#   --scene-text "In the image, there is a tractor in the center. A cow stands in front of the tractor, with a goat on the left of the cow and a sheep on the right of the cow. A pickup truck is located at the back-left of the tractor, and a bulldozer is located at the back-right of the tractor."

# python3 agent_check_pkl0.py \
#   --scene-text "In the image, there is a tractor in the center. A cow stands in front of the tractor, with a goat on the left of the cow and a sheep on the right of the cow. A pickup truck is located at the back-left of the tractor, and a bulldozer is located at the back-right of the tractor." \
#   --scene-pkl inference/saved_scenes/test3.31/example_test3.31_3_fixed.pkl \
#   --output inference/saved_scenes/test3.31/example_test3.31_3_fixed.pkl

# python3 infer20.py \
#   --scene-pkls inference/saved_scenes/test3.31/example_test3.31_3_fixed.pkl \
#   --placeholder-prompt "a photo of PLACEHOLDER, soft daytime lighting, fresh air, a natural and simple rural atmosphere, realistic colors without oversaturation, rich details, and a highly lifelike image."


# demo4
python3 agent_text2pkl0.py \
  --output inference/saved_scenes/test3.31/example_test3.31_5.pkl \
  --scene-text "In the image, there is a jeep in the center. A man stands on the left side of the jeep, and a motorbike is parked on the right side of the jeep. A dog is positioned in front of the jeep, with a fox to the left of the dog and a wolf to the right of the dog."

python3 agent_check_pkl0.py \
  --scene-text "In the image, there is a jeep in the center. A man stands on the left side of the jeep, and a motorbike is parked on the right side of the jeep. A dog is positioned in front of the jeep, with a fox to the left of the dog and a wolf to the right of the dog." \
  --scene-pkl inference/saved_scenes/test3.31/example_test3.31_5.pkl \
  --output inference/saved_scenes/test3.31/example_test3.31_5_fixed.pkl

python3 infer20.py \
  --scene-pkls inference/saved_scenes/test3.31/example_test3.31_5_fixed.pkl \
  --placeholder-prompt "a photo of PLACEHOLDER, cinematic realistic style, clear layers of light and shadow, a slight sense of tension in the air, sharp details, and a realistic scene with strong storytelling."