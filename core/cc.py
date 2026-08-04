"""Curriculum (CC 1.0) th05 helpers"""

import os
import random

import rrr

from .param import CHAR, CFG_FILE, CURRICULUM_CHAR, EXPORT_DIR

"""
    Curriculum Helpers of thrl.
    Copyright (C) 2026  T. Liu and contributors

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


def gen_cfg(playperf):
    """Playperf algorithm. Currently unused.
    :param playperf: rrr.playperf
    :return: A passed config
    :raises OSError: If the configuration file cannot be written
    """
    configs = rrr.playperf(playperf, 5, 100, CHAR)
    cfg = random.choice(configs).cfg if configs else rrr.Cfg05()
    rrr.cfg_write(os.path.join(EXPORT_DIR, CFG_FILE), cfg)
    return cfg


def cfg_to_dict(cfg):
    """Convert an `rrr.Cfg05` obj to dict.
    
    :param cfg
    :return: dic
    """
    return {
        "live": int(cfg.live),
        "bomb": int(cfg.bomb),
        "stg": int(cfg.stg),
        "phase": int(cfg.phase),
        "end": int(cfg.end),
        "cha": int(cfg.cha),
        "rank": int(cfg.rank),
        "power": int(cfg.power),
    }


def cfg_from_dict(d):
    """Reconstruct an `rrr.Cfg05` object from serialized fields.
    
    :param d: if new run.
    :return: Reconstructed configuration, or make_init_cfg()
    """
    if not d:
        return make_init_cfg()
    return rrr.Cfg05(
        live=int(d.get("live", 3)),
        bomb=int(d.get("bomb", 3)),
        stg=int(d.get("stg", 0)),
        phase=int(d.get("phase", 0)),
        end=int(d.get("end", 0)),
        cha=int(d.get("cha", CURRICULUM_CHAR)),
        rank=int(d.get("rank", 0)),
        power=int(d.get("power", 0)),
    )


def cfg_debug(cfg):
    """For print.

    :param cfg: rrr.Cfg05
    :return str
    """
    return (f"cfg05{{lives:{cfg.live}, bombs:{cfg.bomb}, stage:{cfg.stg}, "
            f"phase:{cfg.phase}, end:{cfg.end}, char:{cfg.cha}, "
            f"rank:{cfg.rank}, power:{cfg.power}}}")


def write_cfg(cfg) -> None:
    """Write the active game configuration to export dir.
    
    :param cfg: rrr.Cfg05
    :return: None
    :raises OSError: If the target cannot be written.
    """
    rrr.cfg_write(os.path.join(EXPORT_DIR, CFG_FILE), cfg)


def make_init_cfg():
    # Stage 0 (1) full run, rank 0, default spell (bomb) & live, selected training char.
    """Create the deterministic initial curriculum configuration.
    :return: Initial cfg.
    """
    return rrr.Cfg05(
        live=3,
        bomb=3,
        stg=0,
        phase=0,
        end=0,
        cha=CURRICULUM_CHAR,
        rank=0,
        power=0,
    )


def spec_cfg_inner_gen(cfg):
    """This is for the python side.
    :param cfg: rrr.Cfg05
    :return: rrr.Cfg05
    """
    return rrr.specific_cfg_gen(int(cfg.stg), int(cfg.phase), int(cfg.rank), int(cfg.cha))
