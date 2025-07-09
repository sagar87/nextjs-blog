import Link from "next/link";
import Image from "next/image";

const Header = () => {
  return (
    <header>
      <div className="text-center p-8 my-6 rounded-md border-blue-200 bg-slate-100/25">
        <Image
          src="/logo.png"
          width={60}
          height={60}
          alt="Logo"
          className="mx-auto"
        />
        <h1 className="text-3xl font-bold orange_gradient py-2">
          <Link href="/">Harald Vöhringer, PhD</Link>
        </h1>
        <p className="text-slate-700">Thoughts and observations.</p>
      </div>
    </header>
  );
};

export default Header;
