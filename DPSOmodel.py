
import numpy as np

from utils import get_args
import copy

args = get_args()  # param_parser搜索
class DParticle:
    def __init__(self, length: int, fitness: float = 0):
        self.solution = np.array([0 for _ in range(length)], dtype=int)  # 获得一个长度为size的01随机向量
        self.velocity = np.array([0 for _ in range(length)], dtype=float)  # 获得一个长度为size的零向量
        self.fitness = fitness

    def __str__(self):
        result = [str(e) for e in self.solution]  # 例如['0', '1', '0', '1', '1', '1', '0', '0']
        return '[' + ', '.join(result) + ']'  # 例如'[0,1,0,1,1,1,0,0]'

def calculate_fitness(p, joblist,CP, event,env) -> float:
    rewards_list_Dpso = []
    # print("calculate_fitness",CP)
    env.setCP(CP)
    env.setevent( event)

    # print(len(joblist),len(p))
    # print("----",joblist)

    for i in range(len(joblist)):
        # print(p[i])

        reward_Dpso = env.feedback(joblist[i], p[i], 4)
        rewards_list_Dpso.append(reward_Dpso)

    final_reward_Dpso = sum(rewards_list_Dpso)

    return final_reward_Dpso
weight=0.9
def calculate_fitnessgb(p,joblist, env) -> float:
    rewards_list_Dpso = []
    env.resetdevent()
    print("len",len(joblist),len(p))
    for i in range(500):

        reward_Dpso = env.feedback(joblist[i], p[i], 4)
        rewards_list_Dpso.append(reward_Dpso)



    final_reward_Dpso = sum(rewards_list_Dpso)
    performance_lamda= env.get_total_responseTs(args.Baseline_num, 0)
    performance_success= env.get_totalSuccess(args.Baseline_num, 0)
    performance_util= env.get_avgUtilitizationRate(args.Baseline_num, 0)
    performance_finishT= env.get_totalTimes(args.Baseline_num, 0)
    performance_cost = env.get_totalCost(args.Baseline_num, 0)
    print('avg_responseT:')
    print(performance_lamda)
    print('success_rate:')
    print(performance_success)
    print('utilizationRate:')
    print(performance_util)
    print('finishT:')
    print(performance_finishT)
    print('Cost:')
    print(performance_cost)
    print(final_reward_Dpso)

    return final_reward_Dpso
weight=0.9
class DPSO:
    def __init__(self, joblist ,CP, event, env,population_number=50,times=100, w=0.5682535929979919, c1=1.0517873898303416, c2=1.6161509117073827):

        self.population_number = population_number  # 种群数量
        self.times = times  # 遗传代数
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.vmax=2.303374264117224

        self.cloudlet_num = len(joblist) # 任务数量也就是粒子长度
        self.machine_number = 10  # 机器数量
        # self.chromosome_size = 0  # 染色体长度 int

        self.particles = set()  # Set[Gene]类型
        self.gbest = None
        self.CP=copy.deepcopy(CP)
        self. event = copy.deepcopy( event)
        # print("self.CP", self.CP)
        self.env = env
        self.job =  joblist


    def init_population(self):
        for _ in range(self.population_number):
            size = self.cloudlet_num
            p = DParticle(size)
            p.solution = np.random.randint(0, self.machine_number, size=self.cloudlet_num)


            p.velocity = -5 + 10 * np.random.rand(self.cloudlet_num)
            # print("chromosome:", g.chromosome)
            p.fitness = calculate_fitness(p.solution,self.job,self.CP,self. event,self.env)
            self.particles.add(p)

    def find_best(self):
        best_score = 0
        best_particle = None
        for p in self.particles:
            score = calculate_fitness(p.solution,self.job, self.CP,self. event,self.env)
            if score > best_score:
                best_score = score
                best_particle = p
        return best_particle

    def update_velocity(self, p: DParticle, lb: DParticle, gb: DParticle):
        #功能：生成两个随机数数组 E1 和 E2。
        '''self.cloudlet_num 表示任务的数量（也可能是问题维度）。
        E1 和 E2 是随机系数，用于引入随机性，模拟群体中个体对其他个体之间的影响。
        这两个数组的每个元素都是一个在 0 到 1 之间的随机数。例如，E1 = [0.1, 0.2, 0.002, 0.4, ...]。'''
        E1 = np.array([np.random.random() for _ in range(self.cloudlet_num)])  # [0.1, 0.2, 0.002, 0.4, ...]
        E2 = np.array([np.random.random() for _ in range(self.cloudlet_num)])
        v1 = np.array(gb.solution) - np.array(p.solution)
        v2 = np.array(lb.solution) - np.array(p.solution)
        velocity = self.c1 * E1 * v1 + self.c2 * E2 * v2
        velocity = np.clip(velocity, -self.vmax, self.vmax)
        p.velocity = p.velocity * self.w + velocity
        #print(p.velocity)

    def update_position(self, p: DParticle):
        p.solution = np.abs(p.solution + p.velocity)
        for t in range(len(self.job)):
            if p.solution[t] >= self.machine_number:
                p.solution[t] = np.ceil(p.solution[t]) % self.machine_number
        p.solution = list(map(int, p.solution))

    # 离散粒子群算法
    def execute(self):

        # 初始化基因，基因包含染色体和染色体的适应度
        self.init_population()
        #最好的粒子
        self.gbest = self.find_best()
        # 在某些变体的 PSO 中，例如差分进化粒子群优化（DPSO），
        # 可能会选择种群中的一个粒子作为对比基准，此处移除的粒子lb用作局部最优（local best）
        lb = self.particles.pop()
        # 计算适应度
        best_score = calculate_fitness(lb.solution, self.job,self.CP,self. event,self.env)
        global_best_score = calculate_fitness(self.gbest.solution,self.job, self.CP,self. event,self.env)
        results = []
        # 迭代过程 仅包含速度和位置的更新
        for t in range(self.times):
            #print("t:   ",t)
            for p in self.particles:
                if p is self.gbest:
                    continue
                if p is lb:
                    continue
                self.update_velocity(p, lb, self.gbest)
                self.update_position(p)
                score = calculate_fitness(p.solution,self.job, self.CP,self. event,self.env)
                if score > best_score:
                    best_score = score
                    lb = p
                if score > global_best_score:
                    global_best_score = score
                    self.gbest = p
                    calculate_fitness(p.solution,self.job,self.CP,self. event,self.env)
            results.append(global_best_score)
            if t % 20 == 0:
                print("DPSO iter: ", t, " / ", self.times, ", 适应度: ", global_best_score)
        # return gb.solution
        calculate_fitness(self.gbest.solution,self.job,self.CP, self. event,self.env)
        print(type(self.gbest.solution))
        return self.gbest.solution

    # 输出结果
    def schedule(self):
        result = self.execute()


