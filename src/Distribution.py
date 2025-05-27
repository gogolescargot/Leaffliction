# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Distribution.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ggalon <ggalon@student.42lyon.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/05/27 13:59:07 by ggalon            #+#    #+#              #
#    Updated: 2025/05/27 15:42:14 by ggalon           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
import sys
import matplotlib.pyplot as plt

def distribution():
	if len(sys.argv) != 2:
		print(f"Usage: python {sys.argv[0]} <path>", file=sys.stderr)
		sys.exit(1)

	path = sys.argv[1]

	categories = {}

	for dirpath, _, filenames in os.walk(path):
		if os.path.abspath(dirpath) == os.path.abspath(path):
			continue
		categorie = os.path.basename(dirpath)
		categories[categorie] = len(filenames)

	plt.figure(figsize=(6, 3))

	plt.subplot(131)
	plt.pie(categories.values(), labels=categories.keys())

	plt.subplot(132)
	plt.bar(categories.keys(), categories.values())

	plt.suptitle("Class Distribution")
	plt.show()

	return

if __name__ == "__main__":
	distribution()