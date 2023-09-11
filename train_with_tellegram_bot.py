import gymcad
import torch
import model

import telebot
from telebot import types

bot = telebot.TeleBot('6192192637:AAFoxB7NAeP0ydSK6K1uRlhwLlw1C4yJZzw')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gymcad.BoxCAD(device)
agent1 = model.DQNAgent(env, device)

# MAX_EPISODE = 10
MAX_STEP = 100
BATCH_SIZE = 64
first_step = 4101

safety_flag = False


ckp_path = r'D:\Research\Research\box_cad_tests\4976checkpoint.pt'

agent1.model, agent1.optimizer, start_epoch = model.load_ckp(ckp_path, agent1.model, agent1.optimizer)

def mini_batch_train(loc_bot, loc_message, env,  agent, max_episodes, max_step, batch_size, tar_update_fr, update_fr, eps_step):

  global first_step

  eps_step = first_step
  
  start_traing = False

  global_step = eps_step*max_step
  episode_rewards = []
  losses = []
  difference_nets = []

  agent.target_model.eval()
  agent.model.eval()

  for episode in range(max_episodes):
    state = env.reset()
    if start_traing:
      agent.model.train()
    
    negative_reward_counter = 0
    local_step = 1
    episode_reward = 0

    print_message = ""
    
    while True:
      global_step += 1
      epsi = model.epsilon(global_step)
      action = agent.get_action(state, epsi)
      
      next_state, reward, done, _ = env.step(action)

      agent.reply_buffer.push(state, action, reward, next_state, done)
      episode_reward += reward

      if (len(agent.reply_buffer) > 150) and (global_step % update_fr == 0):
        if start_traing == False:
            agent.model.train()
            start_traing = True
            print("Start_learning")
        agent.update(batch_size)

      negative_reward_counter = negative_reward_counter + 1 if local_step > max_step-25 and reward < 0 else 0

        
      if global_step % tar_update_fr == 0:
         agent.update_target()
      
      if done or local_step >= max_step: 
        episode_rewards.append(episode_reward)
        print_message = "Episode " + str(episode+eps_step) + ": " + str(episode_reward) + " Done: " + str(done) + " step: " + str(local_step)
        print(print_message)
        break

      state = next_state
      local_step += 1

    state = env.reset()
    
    negative_reward_counter = 0
    local_step = 1
    episode_reward_eval = 0
    agent.model.eval()

    if episode%25 == 0:
      checkpoint = {
            'epoch': global_step,
            'state_dict': agent.model.state_dict(),
            'optimizer': agent.optimizer.state_dict()
        }
      model.save_ckp(checkpoint, False, 'box_cad_tests', 'box_cad_tests', str(episode+eps_step))

    if episode%10 == 0:
      loc_bot.send_message(loc_message.from_user.id, print_message, parse_mode='Markdown')

  first_step = episode+eps_step

  return episode_rewards, losses, difference_nets


# episode_rewards1, losses1, difference_nets1 = mini_batch_train(env, agent1, MAX_EPISODE, MAX_STEP, BATCH_SIZE, 25, 1, eps_step=4001)

count_steper = 4001

@bot.message_handler(commands=['start'])
def start(message):

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("Start learning")
    markup.add(btn1)
    bot.send_message(message.from_user.id, "👋 Привет! Я твой бот-помошник!", reply_markup=markup)

@bot.message_handler(content_types=['text'])
def get_text_messages(message):

    if message.text == 'Start learing':
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True) #создание новых кнопок
        btn1 = types.KeyboardButton('100')
        btn2 = types.KeyboardButton('500')
        btn3 = types.KeyboardButton('1000')
        # btn4 = types.KeyboardButton('Stop learing')
        markup.add(btn1, btn2, btn3)
        bot.send_message(message.from_user.id, 'Select the number of eposides', reply_markup=markup) #ответ бота


    elif (message.text == '100'):
        episode_rewards1, losses1, difference_nets1 = mini_batch_train(bot, message, env, agent1, 100, MAX_STEP, BATCH_SIZE, 25, 1, eps_step=4001)
        bot.send_message(message.from_user.id, 'Lerning is completed', parse_mode='Markdown')

    elif (message.text == '500') and (safety_flag == False):
        episode_rewards1, losses1, difference_nets1 = mini_batch_train(bot, message, env, agent1, 500, MAX_STEP, BATCH_SIZE, 25, 1, eps_step=4001)
        bot.send_message(message.from_user.id, 'Lerning is completed', parse_mode='Markdown')
    
    elif (message.text == '1000') and (safety_flag == False):
        episode_rewards1, losses1, difference_nets1 = mini_batch_train(bot, message, env, agent1, 1000, MAX_STEP, BATCH_SIZE, 25, 1, eps_step=4001)
        bot.send_message(message.from_user.id, 'Lerning is completed', parse_mode='Markdown')

    elif message.text == 'Stop learning':
        bot.send_message(message.from_user.id, 'last step is ' + first_step, parse_mode='Markdown')


bot.polling(none_stop=True, interval=0) #обязательная для работы бота часть

