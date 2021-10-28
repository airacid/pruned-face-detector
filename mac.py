from lib.helper.logger import logger
from lib.core.base_trainer.mac_net_work import trainner
import setproctitle



logger.info('train start')
setproctitle.setproctitle("detect")

trainner=trainner()

trainner.train()

