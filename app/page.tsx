import { headers } from "next/headers";
import { detectLanguage, type Lang } from "@/lib/locales";
import { MainPage } from "@/app/components/MainPage";

interface PageProps {
  searchParams?: { lang?: string };
}

export default function Home(props: PageProps) {
  const queryLang = props.searchParams?.lang;
  const acceptLang = headers().get("accept-language");
  const lang: Lang = detectLanguage(queryLang, acceptLang);

  return <MainPage initialLang={lang} />;
}
