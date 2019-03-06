import pygame
import sys
from math import *

#Declaration of constants
dx = 920
dy = 480
rad = 20
gLev = 400

WIN = pi*dx/8+dx/2 - 10
LOSE = -pi*dx/8+dx/2 - 10

red = (255,0,0)
sky = (180,225,255)
earth = (149,69,53)
star = (255,230,20)
green = (0,120,0)
black = (0,0,0)

cvals = [((x*pi/100),cos(x*pi/100)) for x in range(-100,100,1)]
curve = [(x*dx/8 + dx/2,(y-1)*dy/4 + gLev) for (x,y) in cvals]

#initialize physics
ball_pos = 0.0
ball_vel = 0.0
ball_acc = 0.000001
gc = 0.000002

#initialize display
pygame.init()
screen = pygame.display.set_mode((dx,dy))
clock = pygame.time.Clock()

while (True):
	dt = clock.tick(30)

	screen.fill(sky)
	pygame.draw.rect(screen, earth, (0,gLev,dx,dy-gLev), 0)
	pygame.draw.lines(screen, black, False, curve, 3)
	pygame.draw.ellipse(screen, star, (WIN, -dy/2 + gLev - 40 ,20,40), 0)
	pygame.draw.ellipse(screen, red, (LOSE, -dy/2 + gLev - 40 ,20,40), 0)

	#Press left or right, Press ESC to quit
	for event in pygame.event.get():
		if event.type == pygame.KEYDOWN:
			if event.key == pygame.K_ESCAPE:
				pygame.quit()
				sys.exit()

	keys = pygame.key.get_pressed()
	if keys[pygame.K_LEFT] and not keys[pygame.K_RIGHT]:
		ball_vel = ball_vel - ball_acc * dt
	if not keys[pygame.K_LEFT] and keys[pygame.K_RIGHT]:
		ball_vel = ball_vel + ball_acc * dt

	ball_vel = ball_vel - gc*sin(ball_pos) * dt
	ball_vel -= (ball_vel * dt * 0.0001)

	ball_pos = ball_pos + ball_vel * dt
	pygame.draw.circle(screen, green, (int(ball_pos*dx/8 + dx/2), \
								 	 int((cos(ball_pos)-1)*dy/4 + gLev - rad)), rad, 0)

	if ball_pos >= pi:
		print("YOU WON")
		ball_pos = 0.0
		ball_vel = 0.0
		ball_acc = 0.000001
	if ball_pos <= -pi:
		print("YOU LOST")
		ball_pos = 0.0
		ball_vel = 0.0
		ball_acc = 0.000001


	pygame.display.update()













