from hjmodel import HJModel

RES_PATH = '/Users/jameswirth/PycharmProjects/hotjupiter-multiprocess/data/'

if __name__ == '__main__':
    model = HJModel(time=12000, num_systems=250000, res_path=RES_PATH, res_name='example')
    model.run()


