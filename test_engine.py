from app import parse_order_text, product_similarity

def test_free_form():
    text = """Hyndrangea pink 50cm packrate 60 3bxs price 1.65
Hyndrangea julita pink 50cm packrate 60 2bx price 1.65"""
    items, meta = parse_order_text(text)
    assert len(items) == 2
    assert items[0]["boxes"] == 3
    assert items[0]["pack_rate"] == 60
    assert items[0]["unit_price"] == 1.65

def test_table():
    text = """Flower | Length | Pack Rate | Boxes | Total Stems | Unit Price | Amount
Hydrangea Pink | 50cm | 60 | 3 | 180 | 1.65 | 297
Hydrangea Julita Pink | 50cm | 60 | 2 | 120 | 1.65 | 198"""
    items, _ = parse_order_text(text)
    assert len(items) == 2
    assert items[0]["quantity"] == 180
    assert items[1]["total"] == 198

def test_product_similarity():
    assert product_similarity("Hyndrangea pink 50cm", "Hydrangea Pink") > 0.84
    assert product_similarity("Celocia", "Celosia") < 1.0
