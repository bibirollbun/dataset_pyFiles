# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import sys
import os
import time
from datetime import datetime
from bisect import bisect_left, insort
import bisect
import copy
import random
from threading import Thread, Lock
from collections import defaultdict

INPUT_FILE = "/kaggle/input/hashcode-2021-oqr-extension/hashcode.in"
OUTPUT_FILE = "submission.csv"
THREAD_COUNT = 8  # 线程数

class Street:
    def __init__(self, unique_id, start, end, name, length):
        self.UniqueID = unique_id
        self.StartIntersection = start
        self.EndIntersection = end
        self.Name = name
        self.Length = length
        self.IncomingUsageCount = 0
        self.CarsOnStart = 0
        self.IncomingStreetsCount = 0  # 进入该街道的路口的街道数
        self.OutgoingStreetsCount = 0  # 从该街道驶出的路口的街道数

class Car:
    def __init__(self, unique_id, streets):
        self.UniqueID = unique_id
        self.Streets = streets

    def timeNeedToDrive(self):
        return sum(street.Length for street in self.Streets[1:])

class Intersection:
    def __init__(self, id):
        self.ID = id
        self.IncomingStreets = []
        self.OutgoingStreets = []

class CarSimultionPosition:
    def __init__(self, car, time_got_here):
        self.MyCar = car
        self.StreetNumber = 0
        self.TimeGotHere = time_got_here
        self.TimeLeftOnDrive = car.timeNeedToDrive()
        self.StreetLength = [s.Length for s in car.Streets]
        self.StreetUniqueID = [s.UniqueID for s in car.Streets]
        self.StreetEndIntersection = [s.EndIntersection for s in car.Streets]

    def getCurrentStreet(self):
        return self.MyCar.Streets[self.StreetNumber]

    def init(self, time_got_here):
        self.StreetNumber = 0
        self.TimeGotHere = time_got_here
        self.TimeLeftOnDrive = self.MyCar.timeNeedToDrive()

    @staticmethod
    def compareByTimeGotHere(a, b):
        return a.TimeGotHere < b.TimeGotHere

class Problem:
    @staticmethod
    def loadProblem(fileName):
        with open(fileName) as f:
            lines = [l.strip() for l in f if l.strip()]
        
        ptr = 0
        D, I, S, V, F = map(int, lines[ptr].split())
        ptr += 1
        
        problem = Problem()
        problem.Duration = D
        problem.BonusPerCar = F
        problem.Streets = []
        problem.Cars = []
        problem.Intersections = [Intersection(i) for i in range(I)]
        
        streets_map = {}
        for _ in range(S):
            parts = lines[ptr].split()
            B, E, name, L = int(parts[0]), int(parts[1]), parts[2], int(parts[3])
            street = Street(len(problem.Streets), B, E, name, L)
            streets_map[name] = street
            problem.Streets.append(street)
            ptr += 1
            
        for _ in range(V):
            parts = lines[ptr].split()
            P = int(parts[0])
            streets = [streets_map[name] for name in parts[1:P+1]]
            problem.Cars.append(Car(len(problem.Cars), streets))
            ptr += 1
            
        for street in problem.Streets:
            problem.Intersections[street.StartIntersection].OutgoingStreets.append(street)
            problem.Intersections[street.EndIntersection].IncomingStreets.append(street)
            
        for car in problem.Cars:
            for i in range(len(car.Streets) - 1):
                car.Streets[i].IncomingUsageCount += 1
                
        for street in problem.Streets:
            street.CarsOnStart = 0
            
        for car in problem.Cars:
            car.Streets[0].CarsOnStart += 1
        
        # 初始化街道权重相关数据
        for intersection in problem.Intersections:
            for street in intersection.IncomingStreets:
                street.IncomingStreetsCount = len(intersection.IncomingStreets)
            for street in intersection.OutgoingStreets:
                street.OutgoingStreetsCount = len(intersection.OutgoingStreets)
        
        return problem

    def removeUnusedStreets(self):
        count = 0
        for intersection in self.Intersections:
            new_incoming = []
            for street in intersection.IncomingStreets:
                if street.IncomingUsageCount > 0:
                    new_incoming.append(street)
                else:
                    count += 1
            intersection.IncomingStreets = new_incoming
        return count

class GreenLightCycle:
    def __init__(self, street, duration):
        self.MyStreet = street
        self.Duration = duration
        self.GreenLightUsed = False

    def clone(self):
        return GreenLightCycle(self.MyStreet, self.Duration)

class SolutionIntersection:
    def __init__(self, id):
        self.ID = id
        self.GreenLights = []
        self.CurrentGreenLight = 0
        self.CurrentGreenLightChangeTime = 0
        self.LastCarPassTime = -1
        self.GreenLightsArray = []

    def buildGreenLightsArray(self):
        self.GreenLightsArray = []
        for cycle in self.GreenLights:
            self.GreenLightsArray.extend([cycle.MyStreet.UniqueID] * cycle.Duration)

    def hasGreenLights(self):
        return any(g.Duration > 0 for g in self.GreenLights)

    def clone(self):
        cloned = SolutionIntersection(self.ID)
        cloned.GreenLights = [g.clone() for g in self.GreenLights]
        return cloned

class Solution:
    def __init__(self, num_intersections):
        self.Intersections = [SolutionIntersection(i) for i in range(num_intersections)]

    def countIntersectionsWithGreenLights(self):
        return sum(1 for i in self.Intersections if i.hasGreenLights())

    def clone(self):
        cloned = Solution(len(self.Intersections))
        cloned.Intersections = [inter.clone() for inter in self.Intersections]
        return cloned

def initBasicSolution(problem, solution):
    for intersection in problem.Intersections:
        sol_inter = solution.Intersections[intersection.ID]
        sol_inter.GreenLights.clear()
        for street in intersection.IncomingStreets:
            duration = max(1, street.CarsOnStart)
            sol_inter.GreenLights.append(GreenLightCycle(street, duration))
    return solution

def calculate_street_weight(street, vehicle_state, a=0.182, b=0.046, c=0.81, d=0.025):
    """
    计算街道的权重
    :param street: Street 对象
    :param vehicle_state: 当前车辆的状态（字典，键为车辆ID，值为车辆状态）
    :param a, b, c, d: 权重系数
    :return: 综合权重
    """
    # 1. 街道长度权重（越长权重越高）
    length_weight = street.Length
    
    # 2. 路口相关权重
    # 进入路口的街道数（越多权重越低）
    incoming_weight = 1 / (street.IncomingStreetsCount + 1)
    # 驶出路口的街道数（越多权重越高）
    outgoing_weight = street.OutgoingStreetsCount
    
    # 3. 车辆进度权重（所有车辆进度之和）
    progress_weight = sum(
        vehicle['onRoadSecond'] / street.Length
        for vehicle in vehicle_state.values()
        if vehicle['arr'][vehicle['curIndex']] == street.Name
    )
    
    # 综合权重
    total_weight = a * length_weight + b * incoming_weight + c * outgoing_weight + d * progress_weight
    return total_weight

def optimizeGreenLightOrder4(problem, solution):
    score = 0
    current_time = 0

    all_sim_cars = []
    car_queue = defaultdict(list)

    simulation_start = - (len(problem.Cars) + 1)
    for car in problem.Cars:
        sim_car = CarSimultionPosition(car, simulation_start)
        all_sim_cars.append(sim_car)
        end_inter = car.Streets[0].EndIntersection
        insort(car_queue[end_inter], sim_car, key=lambda x: x.TimeGotHere)
        simulation_start += 1

    # 初始化车辆状态
    vehicle_state = {car.UniqueID: {'onRoadSecond': 0, 'arr': [s.Name for s in car.Streets], 'curIndex': 0} for car in problem.Cars}

    # 计算每个街道的权重
    street_weights = {street.Name: calculate_street_weight(street, vehicle_state) for street in problem.Streets}

    while current_time <= problem.Duration:
        for inter in solution.Intersections:
            if not inter.GreenLights:
                continue

            if inter.CurrentGreenLightChangeTime <= current_time:
                inter.CurrentGreenLight = (inter.CurrentGreenLight + 1) % len(inter.GreenLights)
                inter.CurrentGreenLightChangeTime = current_time + inter.GreenLights[inter.CurrentGreenLight].Duration

            q = car_queue[inter.ID]
            if not inter.GreenLights[inter.CurrentGreenLight].GreenLightUsed:
                first_cars = {}
                i = 0
                while i < len(q):
                    car_pos = q[i]
                    if car_pos.TimeGotHere > current_time:
                        break
                    street_id = car_pos.getCurrentStreet().UniqueID
                    if street_id not in first_cars:
                        first_cars[street_id] = car_pos
                    i += 1

                # 根据街道权重排序
                sorted_cars = sorted(first_cars.values(), key=lambda x: street_weights[x.getCurrentStreet().Name], reverse=True)

                for car_pos in sorted_cars:
                    street = car_pos.getCurrentStreet()
                    req_green = -1
                    for g_idx, g in enumerate(inter.GreenLights):
                        if g.MyStreet.UniqueID == street.UniqueID:
                            req_green = g_idx
                            break
                    if req_green == -1:
                        continue

                    if inter.GreenLights[req_green].GreenLightUsed:
                        continue
                    if inter.GreenLights[req_green].Duration != inter.GreenLights[inter.CurrentGreenLight].Duration:
                        continue

                    inter.GreenLights[req_green], inter.GreenLights[inter.CurrentGreenLight] = \
                        inter.GreenLights[inter.CurrentGreenLight], inter.GreenLights[req_green]
                    inter.GreenLights[inter.CurrentGreenLight].GreenLightUsed = True
                    break

            car_passed = False
            i = 0
            while i < len(q):
                car_pos = q[i]
                if car_pos.TimeGotHere > current_time:
                    break
                if car_passed:
                    i += 1
                    continue
                if car_pos.getCurrentStreet().UniqueID != inter.GreenLights[inter.CurrentGreenLight].MyStreet.UniqueID:
                    i += 1
                    continue

                car_passed = True
                inter.GreenLights[inter.CurrentGreenLight].GreenLightUsed = True

                car_pos.StreetNumber += 1
                new_street = car_pos.getCurrentStreet()
                car_pos.TimeGotHere = current_time + new_street.Length
                car_pos.TimeLeftOnDrive -= new_street.Length

                del q[i]

                if car_pos.StreetNumber == len(car_pos.MyCar.Streets) - 1:
                    if car_pos.TimeGotHere <= problem.Duration:
                        score += problem.BonusPerCar + (problem.Duration - car_pos.TimeGotHere)
                else:
                    end_inter = new_street.EndIntersection
                    insort(car_queue[end_inter], car_pos, key=lambda x: x.TimeGotHere)
        current_time += 1

    return score

def runSimulationLite(problem, solution, car_positions=None):
    if car_positions is None:
        car_positions = [CarSimultionPosition(car, 0) for car in problem.Cars]
        car_positions.sort(key=lambda x: x.TimeGotHere)
        return runSimulationLite(problem, solution, car_positions)

    score = 0
    current_time = 0
    n = len(car_positions)

    for inter in solution.Intersections:
        inter.buildGreenLightsArray()
        inter.LastCarPassTime = -1

    while current_time <= problem.Duration:
        i = 0
        while i < n:
            car_pos = car_positions[i]
            if car_pos.TimeGotHere > current_time:
                break

            inter = solution.Intersections[car_pos.StreetEndIntersection[car_pos.StreetNumber]]
            if inter.LastCarPassTime == current_time:
                i += 1
                continue

            if not inter.GreenLightsArray:
                i += 1
                continue

            cycle_idx = current_time % len(inter.GreenLightsArray)
            current_green_id = inter.GreenLightsArray[cycle_idx]
            if car_pos.StreetUniqueID[car_pos.StreetNumber] != current_green_id:
                i += 1
                continue

            inter.LastCarPassTime = current_time
            car_pos.StreetNumber += 1
            new_length = car_pos.StreetLength[car_pos.StreetNumber]
            car_pos.TimeGotHere = current_time + new_length
            car_pos.TimeLeftOnDrive -= new_length

            if car_pos.StreetNumber == len(car_pos.MyCar.Streets) - 1:
                if car_pos.TimeGotHere <= problem.Duration:
                    score += problem.BonusPerCar + (problem.Duration - car_pos.TimeGotHere)
                car_positions.pop(i)
                n -= 1
            else:
                new_pos = bisect.bisect_left(car_positions, car_pos.TimeGotHere, i + 1, n, key=lambda x: x.TimeGotHere)
                car_positions.insert(new_pos, car_positions.pop(i))
                i -= 1
            i += 1
        current_time += 1
    return score

class MyRandom:
    def __init__(self, seed):
        self.state = seed

    def nextInt(self, n):
        self.state ^= (self.state << 3) & 0xFFFFFFFFFFFFFFFF
        self.state ^= (self.state >> 1)
        self.state ^= (self.state << 11) & 0xFFFFFFFFFFFFFFFF
        return abs(self.state) % n

    def nextLong(self, n):
        return self.nextInt(n)

class BruteForceSwap:
    def __init__(self):
        self.lock = Lock()
        self.best_solution = None
        self.best_score = 0
        self.optimization_start = datetime.now()

    def optimize(self, problem, solution):
        self.best_solution = solution.clone()
        self.best_score = runSimulationLite(problem, solution)

        threads = []
        for _ in range(THREAD_COUNT):
            t = Thread(target=self.optimize_thread, args=(problem,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        return self.best_solution

    def optimize_thread(self, problem):
        local_solution = self.best_solution.clone()
        local_score = self.best_score
        rand = MyRandom(random.randint(1, 1000000))

        car_positions = [CarSimultionPosition(car, 0) for car in problem.Cars]
        car_positions.sort(key=lambda x: x.TimeGotHere)
        tmp_positions = [None] * len(car_positions)

        total_weight = sum(len(inter.GreenLights) ** 2 for inter in local_solution.Intersections)
        cum_weights = []
        s = 0
        for inter in local_solution.Intersections:
            cum_weights.append(s)
            s += len(inter.GreenLights) ** 2

        for _ in range(20000):
            r = rand.nextLong(total_weight)
            inter_idx = bisect.bisect_left(cum_weights, r)
            inter = local_solution.Intersections[inter_idx]

            if len(inter.GreenLights) < 2:
                continue

            action = rand.nextInt(2)
            if action == 0:  # 交换绿灯顺序
                a = rand.nextInt(len(inter.GreenLights))
                b = rand.nextInt(len(inter.GreenLights))
                while a == b:
                    b = rand.nextInt(len(inter.GreenLights))
                inter.GreenLights[a], inter.GreenLights[b] = inter.GreenLights[b], inter.GreenLights[a]
            else:  # 调整绿灯持续时间
                idx = rand.nextInt(len(inter.GreenLights))
                delta = rand.nextInt(5) - 2  # 随机调整 -2 到 +2
                inter.GreenLights[idx].Duration = max(1, inter.GreenLights[idx].Duration + delta)

            new_score = runSimulationLite(problem, local_solution, car_positions)
            if new_score > local_score:
                local_score = new_score
                with self.lock:
                    if new_score > self.best_score:
                        self.best_score = new_score
                        self.best_solution = local_solution.clone()

# 主程序
if __name__ == "__


