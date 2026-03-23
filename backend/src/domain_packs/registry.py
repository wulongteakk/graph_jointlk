from src.domain_packs.construction.pack import ConstructionPack


def get_domain_pack(pack_id: str = "construction"):
    if pack_id == "construction":
        return ConstructionPack()
    raise ValueError(f"Unknown pack_id: {pack_id}")